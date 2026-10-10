// MobileGlues - gl/gl.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "../includes.h"
#include <GL/gl.h>
#include <cmath>
#include <cstdlib>
#include "glcorearb.h"
#include "log.h"
#include "../gles/loader.h"
#include "../config/settings.h"
#include "mg.h"
#include "framebuffer.h"
#include "../egl/context.h"
#include <mutex>
#include <memory>
#include <ska/flat_hash_map.hpp>

#define DEBUG 0

static GLclampd currentDepthValue;

thread_local GLenum mg_clip_origin = GL_LOWER_LEFT;
thread_local GLenum mg_clip_depth_mode = GL_NEGATIVE_ONE_TO_ONE;

bool mg_modern_depth_compat_enabled() {
    const char* v = std::getenv("DROIDBRIDGE_MOBILEGLUES_ZERO_TO_ONE_DEPTH");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

#include "framebuffer.h"

void glClearDepth(GLclampd depth) {
    LOG()
    currentDepthValue = depth;
    GLES.glClearDepthf((float)depth);
    CHECK_GL_ERROR
}

void glDepthRange(GLclampd near_val, GLclampd far_val) {
    LOG()
    // Keep 26.1.2 and older on MobileGlues' original behavior. Minecraft 26.2+
    // changed to the modern reverse-Z path and is opted in explicitly by DroidBridge.
    if (!mg_modern_depth_compat_enabled()) return;

    GLES.glDepthRangef(static_cast<GLfloat>(near_val), static_cast<GLfloat>(far_val));
    CHECK_GL_ERROR
}

namespace {
using mg_clip_control_proc = void (*)(GLenum origin, GLenum depth);
using mg_egl_generic_proc = void (*)();
using mg_egl_get_proc_address_proc = mg_egl_generic_proc (*)(const char*);

mg_clip_control_proc resolve_gles_clip_control() {
    // Resolve from the GLES implementation first so we never recurse into
    // MobileGlues' own exported desktop symbol.
    mg_clip_control_proc fn = nullptr;
    if (gles != nullptr) {
        fn = reinterpret_cast<mg_clip_control_proc>(proc_address(gles, "glClipControlEXT"));
    }

    // Android vendors are allowed to expose extension entry points only through
    // eglGetProcAddress. Ask the backend EGL directly, not MobileGlues'
    // eglGetProcAddress wrapper, to keep the call in the same driver/context.
    if (fn == nullptr && egl != nullptr) {
        auto backend_get_proc = reinterpret_cast<mg_egl_get_proc_address_proc>(
            proc_address(egl, "eglGetProcAddress"));
        if (backend_get_proc != nullptr) {
            fn = reinterpret_cast<mg_clip_control_proc>(backend_get_proc("glClipControlEXT"));
        }
    }
    return fn;
}
} // namespace

void glClipControl(GLenum origin, GLenum depth) {
    LOG()

    // Pre-26.2 MobileGlues historically exported this as a no-op. Preserve that
    // behavior so older working versions cannot inherit 26.2 reverse-Z state.
    if (!mg_modern_depth_compat_enabled()) return;

    if ((origin != GL_LOWER_LEFT && origin != GL_UPPER_LEFT) ||
        (depth != GL_NEGATIVE_ONE_TO_ONE && depth != GL_ZERO_TO_ONE)) {
        mg_set_gl_error(GL_INVALID_ENUM);
        return;
    }

    static mg_clip_control_proc clip_control = resolve_gles_clip_control();
    if (clip_control == nullptr) {
        if (origin == GL_LOWER_LEFT && depth == GL_NEGATIVE_ONE_TO_ONE) {
            mg_clip_origin = origin;
            mg_clip_depth_mode = depth;
            return;
        }
        mg_set_gl_error(GL_INVALID_OPERATION);
        return;
    }

    clip_control(origin, depth);
    mg_clip_origin = origin;
    mg_clip_depth_mode = depth;
    static thread_local int db_clip_log = 0;
    if (db_clip_log++ < 16) {
        LOG_W_FORCE("[DroidBridge modern depth] glClipControl origin=0x%x depth=0x%x -> GL_EXT_clip_control",
                    origin, depth)
    }
    CHECK_GL_ERROR
}

// The program, VAO and VBO belong to the context that created them. They used to
// be three process-wide names behind an `if (program) return;` guard, so once the
// application destroyed and recreated its EGL context the guard still saw
// non-zero names and skipped re-creation for the rest of the process --
// DrawDepthClearTri went on binding names the new context never made and the
// ANGLE depth-clear workaround silently stopped doing anything.
//
// Kept per context rather than invalidated on switch: invalidating would make two
// alternating contexts recompile and relink the program on every switch, and
// orphan the previous one each time. Nothing is deleted when an entry is dropped,
// for the reason multidraw_check_context() gives -- the objects belong to a
// context that may already be gone, and deleting them against whichever context
// is current now would delete somebody else's names.
namespace {
struct depth_clear_objects_t {
    GLuint program = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
};
std::mutex g_depthClearMutex;
// Held by pointer, because the reference this table hands out outlives the lock.
// A caller keeps the depth_clear_objects_t& across GL calls while another thread
// can be adding its own context, and the map moves its elements when it grows.
// The unique_ptr is what stays put; the map is free to rehash around it.
ska::flat_hash_map<unsigned long long, std::unique_ptr<depth_clear_objects_t>> g_depthClearCtxs;
depth_clear_objects_t g_depthClearDefault;

depth_clear_objects_t& depth_clear_objects() {
    const unsigned long long cur = g_current_ctx ? g_current_ctx->id : 0;
    if (cur == 0) return g_depthClearDefault;
    std::lock_guard<std::mutex> lock(g_depthClearMutex);
    std::unique_ptr<depth_clear_objects_t>& slot = g_depthClearCtxs[cur];
    if (!slot) slot = std::make_unique<depth_clear_objects_t>();
    return *slot;
}
} // namespace

void mg_depth_clear_forget_context(unsigned long long ctx_id) {
    if (ctx_id == 0) return;
    std::lock_guard<std::mutex> lock(g_depthClearMutex);
    g_depthClearCtxs.erase(ctx_id);
}

static const GLfloat kFullScreenTri[3][2] = {{-1.0f, -1.0f}, {3.0f, -1.0f}, {-1.0f, 3.0f}};

static const char* kDepthClearVS = R"glsl(
    #version 300 es
    layout(location = 0) in vec2 aPos;
    void main() {
        // Write far‐plane depth
        gl_Position = vec4(aPos, 1.0, 1.0);
    }
)glsl";
static const char* kDepthClearFS = R"glsl(
    #version 300 es
    precision mediump float;
    out vec4 fragColor;
    void main() {
        // Empty—color writes will be disabled
        fragColor = vec4(0.0);
    }
)glsl";

void InitDepthClearCoreProfile() {
    depth_clear_objects_t& obj = depth_clear_objects();
    if (obj.program) return;

    auto compile = [&](GLenum type, const char* src) {
        GLuint s = GLES.glCreateShader(type);
        GLES.glShaderSource(s, 1, &src, nullptr);
        GLES.glCompileShader(s);
        return s;
    };
    GLuint vs = compile(GL_VERTEX_SHADER, kDepthClearVS);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kDepthClearFS);

    obj.program = GLES.glCreateProgram();
    GLES.glAttachShader(obj.program, vs);
    GLES.glAttachShader(obj.program, fs);
    GLES.glLinkProgram(obj.program);
    GLES.glDeleteShader(vs);
    GLES.glDeleteShader(fs);

    GLES.glGenVertexArrays(1, &obj.vao);
    GLES.glGenBuffers(1, &obj.vbo);

    GLES.glBindVertexArray(obj.vao);
    GLES.glBindBuffer(GL_ARRAY_BUFFER, obj.vbo);
    GLES.glBufferData(GL_ARRAY_BUFFER, sizeof(kFullScreenTri), kFullScreenTri, GL_STATIC_DRAW);

    GLES.glEnableVertexAttribArray(0);
    GLES.glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

    GLES.glBindBuffer(GL_ARRAY_BUFFER, 0);
    GLES.glBindVertexArray(0);
}

void DrawDepthClearTri() {
    InitDepthClearCoreProfile();
    depth_clear_objects_t& obj = depth_clear_objects();

    GLboolean prevColorMask[4];
    GLES.glGetBooleanv(GL_COLOR_WRITEMASK, prevColorMask);
    GLboolean prevDepthMask;
    GLES.glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
    GLint prevDepthFunc;
    GLES.glGetIntegerv(GL_DEPTH_FUNC, &prevDepthFunc);

    GLES.glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    GLES.glDepthMask(GL_TRUE);
    GLES.glDepthFunc(GL_ALWAYS);

    GLES.glUseProgram(obj.program);
    GLES.glBindVertexArray(obj.vao);
    GLES.glDrawArrays(GL_TRIANGLES, 0, 3);
    GLES.glBindVertexArray(0);
    GLES.glUseProgram(0);

    GLES.glDepthFunc(prevDepthFunc);
    GLES.glDepthMask(prevDepthMask);
    GLES.glColorMask(prevColorMask[0], prevColorMask[1], prevColorMask[2], prevColorMask[3]);
}

void glClear(GLbitfield mask) {
    LOG();
    LOG_D("glClear, mask = 0x%x", mask);

    INIT_CHECK_GL_ERROR

    CHECK_GL_ERROR_NO_INIT

    if (global_settings.angle == AngleMode::Enabled && mask == GL_DEPTH_BUFFER_BIT &&
        std::fabs(currentDepthValue - 1.0f) <= 0.001f && mg_draw_framebuffer_all_none()) {
        LOG_D("doing depth workaround")
        if (global_settings.angle_depth_clear_fix_mode == AngleDepthClearFixMode::Mode1)
            // Workaround for ANGLE depth-clear bug: if depth≈1.0, draw a fullscreen triangle at z=1.0 to force actual
            // depth buffer write.
            DrawDepthClearTri();
        else if (global_settings.angle_depth_clear_fix_mode == AngleDepthClearFixMode::Mode2) {
            // Or just explicitly clear depth buffer and see what's happened
            const GLfloat clear_depth_value = 1.0f;
            GLES.glClearBufferfv(GL_DEPTH, 0, &clear_depth_value);
        }
        // Clear again
        GLES.glClear(mask);
    } else {
        GLES.glClear(mask);
    }

    CHECK_GL_ERROR_NO_INIT;
}

void glHint(GLenum target, GLenum mode) {
    LOG()
    LOG_D("glHint, target = %s, mode = %s", glEnumToString(target), glEnumToString(mode))
}
