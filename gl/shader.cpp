// MobileGlues - gl/shader.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include <cctype>
#include <cstdlib>
#include <cstring>
#include "shader.h"

#include <GL/gl.h>
#include "log.h"
#include "program.h"
#include "../gles/loader.h"
#include "../includes.h"
#include "glsl/glsl_for_es.h"
#include "../config/settings.h"
#include "FSR1/FSR1.h"

#define DEBUG 0

struct shader_t shaderInfo;

UnorderedMap<GLuint, bool> shader_map_is_sampler_buffer_emulated;
// If the Mali dynamic output-array read rewrite produces GLSL the driver rejects,
// retain the store-only translation so glGetShaderiv can retry compilation safely.
static UnorderedMap<GLuint, std::string> shader_map_mali_store_only_fallback;

bool can_run_essl3(unsigned int esversion, const char* glsl) {
    if (strncmp(glsl, "#version 100", 12) == 0) {
        return true;
    }

    unsigned int glsl_version = 0;
    if (strncmp(glsl, "#version 300 es", 15) == 0) {
        glsl_version = 300;
    } else if (strncmp(glsl, "#version 310 es", 15) == 0) {
        glsl_version = 310;
    } else if (strncmp(glsl, "#version 320 es", 15) == 0) {
        glsl_version = 320;
    } else {
        return false;
    }
    return esversion >= glsl_version;
}

bool is_direct_shader(const char* glsl) {
    bool es3_ability = can_run_essl3(hardware->es_version, glsl);
    return es3_ability;
}

bool check_if_sampler_buffer_used(std::string str) {
    return str.find("samplerBuffer") != std::string::npos;
}

namespace {
bool droidbridge_zero_to_one_depth_compat_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("DROIDBRIDGE_MOBILEGLUES_ZERO_TO_ONE_DEPTH");
        if (value == nullptr || value[0] == '\0') return false;
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "off") != 0 &&
               std::strcmp(value, "OFF") != 0;
    }();
    return enabled;
}

bool droidbridge_backend_has_ext_clip_control() {
    static const bool available = []() {
        if (GLES.glGetIntegerv == nullptr || GLES.glGetStringi == nullptr) return false;
        GLint count = 0;
        GLES.glGetIntegerv(GL_NUM_EXTENSIONS, &count);
        for (GLint i = 0; i < count; ++i) {
            const GLubyte* ext = GLES.glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i));
            if (ext != nullptr && std::strcmp(reinterpret_cast<const char*>(ext), "GL_EXT_clip_control") == 0)
                return true;
        }
        return false;
    }();
    return available;
}

bool droidbridge_is_dh_terrain_vertex_shader(const std::string& src) {
    return src.find("uniform mat4 uCombinedMatrix") != std::string::npos &&
           src.find("uniform float uWorldYOffset") != std::string::npos &&
           src.find("uniform float uEarthRadius") != std::string::npos &&
           src.find("uniform float uMircoOffset") != std::string::npos &&
           src.find("gl_Position = uCombinedMatrix") != std::string::npos;
}

bool droidbridge_dh_depth_already_remapped(const std::string& src) {
    return src.find("DroidBridge/MobileGlues: DH 26.2+ ZERO_TO_ONE -> GLES clip Z") != std::string::npos ||
           src.find("gl_Position.z = (2.0 * gl_Position.z) - gl_Position.w") != std::string::npos ||
           src.find("gl_Position.z = 2.0 * gl_Position.z - gl_Position.w") != std::string::npos;
}

// Flashback uses the conventional Dear ImGui shader interface.  On MobileGlues
// 2.x the font atlas reaches GLES as RGBA, but the observed atlas samples have
// valid alpha with zero RGB.  The regular ImGui shader multiplies both RGB and
// alpha, turning every otherwise-correct UI colour black while preserving the
// UI's transparency.  Treat zero-RGB/nonzero-alpha texels as an alpha mask,
// which is the semantic the font atlas expects, while leaving genuinely coloured
// ImGui textures alone.
bool droidbridge_is_imgui_vertex_shader(const std::string& src) {
    return src.find("ProjMtx") != std::string::npos &&
           src.find("Position") != std::string::npos &&
           src.find("UV") != std::string::npos &&
           src.find("Color") != std::string::npos &&
           src.find("Frag_UV") != std::string::npos &&
           src.find("Frag_Color") != std::string::npos;
}

bool droidbridge_is_imgui_fragment_shader(const std::string& src) {
    return src.find("Texture") != std::string::npos &&
           src.find("Frag_UV") != std::string::npos &&
           src.find("Frag_Color") != std::string::npos &&
           (src.find("Out_Color") != std::string::npos || src.find("gl_FragColor") != std::string::npos);
}

std::string droidbridge_imgui_alpha_mask_essl300(GLenum shader_type) {
    if (shader_type == GL_VERTEX_SHADER) {
        return R"MGIMGUI(#version 300 es
precision highp float;
precision highp int;
uniform mat4 ProjMtx;
in vec2 Position;
in vec2 UV;
in vec4 Color;
out vec2 Frag_UV;
out vec4 Frag_Color;
void main() {
    Frag_UV = UV;
    Frag_Color = Color;
    gl_Position = ProjMtx * vec4(Position.xy, 0.0, 1.0);
}
)MGIMGUI";
    }
    if (shader_type == GL_FRAGMENT_SHADER) {
        return R"MGIMGUI(#version 300 es
precision mediump float;
precision mediump int;
uniform sampler2D Texture;
in vec2 Frag_UV;
in vec4 Frag_Color;
layout(location = 0) out vec4 Out_Color;
void main() {
    vec4 texel = texture(Texture, Frag_UV.st);
    float maxRgb = max(max(texel.r, texel.g), texel.b);
    vec3 sampleRgb = (maxRgb <= (1.0 / 255.0) && texel.a > 0.0) ? vec3(1.0) : texel.rgb;
    Out_Color = Frag_Color * vec4(sampleRgb, texel.a);
}
)MGIMGUI";
    }
    return {};
}

bool droidbridge_patch_dh_zero_to_one_vertex_depth_fallback(std::string& src) {
    if (!droidbridge_zero_to_one_depth_compat_enabled()) return false;
    if (droidbridge_backend_has_ext_clip_control()) return false;
    if (!droidbridge_is_dh_terrain_vertex_shader(src)) return false;
    if (droidbridge_dh_depth_already_remapped(src)) return false;

    const std::size_t assignment = src.find("gl_Position = uCombinedMatrix");
    if (assignment == std::string::npos) return false;
    const std::size_t semi = src.find(';', assignment);
    if (semi == std::string::npos) return false;

    // Fallback for GLES implementations without GL_EXT_clip_control. DH 26.2+
    // builds a ZERO_TO_ONE projection. GLES' default clip convention is -1..1,
    // so remap clip-space Z once before the viewport transform.
    const std::string remap =
        "\n    gl_Position.z = (2.0 * gl_Position.z) - gl_Position.w;"
        " // DroidBridge/MobileGlues: DH 26.2+ ZERO_TO_ONE -> GLES clip Z";
    src.insert(semi + 1, remap);
    return true;
}
} // namespace

void glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length) {
    LOG()
    shaderInfo.id = 0;
    shaderInfo.converted = "";
    shaderInfo.frag_data_changed_converted.clear();
    shaderInfo.frag_data_changed = 0;
    size_t l = 0;
    for (int i = 0; i < count; i++)
        l += (length && length[i] >= 0) ? length[i] : strlen(string[i]);
    std::string glsl_src, essl_src;
    glsl_src.reserve(l + 1);
    if (length) {
        for (int i = 0; i < count; i++) {
            if (length[i] >= 0)
                glsl_src += std::string_view(string[i], length[i]);
            else
                glsl_src += string[i];
        }
    } else {
        for (int i = 0; i < count; i++) {
            glsl_src += string[i];
        }
    }

    GLint shaderType = 0;
    GLES.glGetShaderiv(shader, GL_SHADER_TYPE, &shaderType);
    if (shaderType == GL_VERTEX_SHADER &&
        droidbridge_patch_dh_zero_to_one_vertex_depth_fallback(glsl_src)) {
        static thread_local int droidbridge_dh_depth_fallback_log = 0;
        if (droidbridge_dh_depth_fallback_log++ < 8) {
            LOG_W_FORCE("[DH26.2 CLIP FIX v2] GL_EXT_clip_control unavailable; applied one-time DH terrain ZERO_TO_ONE shader fallback shader=%u",
                        shader)
        }
    }


    bool is_sampler_buffer_emulated = hardware->emulate_texture_buffer && check_if_sampler_buffer_used(glsl_src);

    const bool droidbridge_imgui_shader =
        (shaderType == GL_VERTEX_SHADER && droidbridge_is_imgui_vertex_shader(glsl_src)) ||
        (shaderType == GL_FRAGMENT_SHADER && droidbridge_is_imgui_fragment_shader(glsl_src));

    if (droidbridge_imgui_shader && hardware->es_version >= 300) {
        shader_map_mali_store_only_fallback.erase(shader);
        essl_src = droidbridge_imgui_alpha_mask_essl300(shaderType);
        static bool logged_vertex = false;
        static bool logged_fragment = false;
        bool* logged = shaderType == GL_VERTEX_SHADER ? &logged_vertex : &logged_fragment;
        if (!*logged) {
            *logged = true;
            LOG_W_FORCE("[MobileGlues][ImGui alpha-mask compat] installed %s shader %u; zero-RGB atlas texels use alpha-mask semantics",
                        shaderType == GL_VERTEX_SHADER ? "vertex" : "fragment", shader)
        }
    } else if (is_direct_shader(glsl_src.c_str())) {
        shader_map_mali_store_only_fallback.erase(shader);
        LOG_D("[INFO] [Shader] Direct shader source: ")
        LOG_D("%s", glsl_src.c_str())
        essl_src = glsl_src;
    } else {
        int glsl_version = getGLSLVersion(glsl_src.c_str());
        LOG_D("[INFO] [Shader] Shader source: ")
        LOG_D("%s", glsl_src.c_str())
        GLint shaderType;
        GLES.glGetShaderiv(shader, GL_SHADER_TYPE, &shaderType);
        int return_code = 0;
        std::string mali_store_only_fallback;
        essl_src = GLSLtoGLSLES(glsl_src.c_str(), shaderType, hardware->es_version, glsl_version, return_code,
                                &mali_store_only_fallback);
        if (!mali_store_only_fallback.empty())
            shader_map_mali_store_only_fallback[shader] = mali_store_only_fallback;
        else
            shader_map_mali_store_only_fallback.erase(shader);

        if (essl_src.empty()) {
            LOG_E("Failed to convert shader %d.", shader)
            return;
        }
        LOG_D("\n[INFO] [Shader] Converted Shader source: \n%s", essl_src.c_str())
    }
    if (!essl_src.empty()) {
        shaderInfo.id = shader;
        shaderInfo.converted = essl_src;
        const char* s[] = {essl_src.c_str()};
        // MobileGlues concatenates all incoming source chunks into essl_src.
        // Forward exactly one string; using the original count with a one-entry
        // array can read past s[] when the caller supplied multiple chunks.
        GLES.glShaderSource(shader, 1, s, nullptr);
        if (hardware->emulate_texture_buffer)
            shader_map_is_sampler_buffer_emulated[shader] = is_sampler_buffer_emulated;
    } else
        LOG_E("Failed to convert glsl.")
    CHECK_GL_ERROR
}

void glGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    LOG()
    GLES.glGetShaderiv(shader, pname, params);

    if (pname == GL_COMPILE_STATUS && !*params) {
        auto fallback = shader_map_mali_store_only_fallback.find(shader);
        if (fallback != shader_map_mali_store_only_fallback.end() && !fallback->second.empty()) {
            GLchar firstLog[512] = {};
            GLES.glGetShaderInfoLog(shader, 512, nullptr, firstLog);

            const GLchar* fallbackSource = fallback->second.c_str();
            GLES.glShaderSource(shader, 1, &fallbackSource, nullptr);
            GLES.glCompileShader(shader);

            GLint retryStatus = GL_FALSE;
            GLES.glGetShaderiv(shader, GL_COMPILE_STATUS, &retryStatus);
            if (retryStatus == GL_TRUE) {
                *params = GL_TRUE;
                if (shaderInfo.id == shader) {
                    shaderInfo.converted = fallback->second;
                    shaderInfo.frag_data_changed_converted.clear();
                    shaderInfo.frag_data_changed = 0;
                }
                LOG_W_FORCE("[Shader] Mali dynamic-read rewrite rejected; store-only retry succeeded for shader %u. First error: %s",
                            shader, firstLog)
                shader_map_mali_store_only_fallback.erase(fallback);
                CHECK_GL_ERROR
                return;
            }

            GLchar retryLog[512] = {};
            GLES.glGetShaderInfoLog(shader, 512, nullptr, retryLog);
            LOG_W_FORCE("[Shader] Mali store-only retry also failed for shader %u. read-rewrite=%s store-only=%s",
                        shader, firstLog, retryLog)
            shader_map_mali_store_only_fallback.erase(fallback);
        }
    }

    if (global_settings.ignore_error >= IgnoreErrorLevel::Partial && pname == GL_COMPILE_STATUS && !*params) {
        GLchar infoLog[512];
        GLES.glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        LOG_W_FORCE("Shader %d compilation failed: \n%s", shader, infoLog)
        LOG_W_FORCE("Now try to cheat.")
        *params = GL_TRUE;
    }
    CHECK_GL_ERROR
}

GLuint glCreateShader(GLenum shaderType) {
    if (global_settings.fsr1_setting != FSR1_Quality_Preset::Disabled && !fsrInitialized) {
        InitFSRResources();
    }

    LOG()
    LOG_D("glCreateShader(%s)", glEnumToString(shaderType))
    GLuint shader = GLES.glCreateShader(shaderType);
    if (shader != 0 && hardware->emulate_texture_buffer) shader_map_is_sampler_buffer_emulated[shader] = false;
    CHECK_GL_ERROR
    return shader;
}