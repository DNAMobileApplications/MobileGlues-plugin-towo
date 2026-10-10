// MobileGlues - gl/drawing.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header

#include "drawing.h"
#include "enable.h"
#include <cstdlib>
#include <cstring>
#include "restart.h"
#include "buffer.h"
#include "framebuffer.h"
#include "mg.h"
#include "texture.h"
#include "../egl/context.h"

#define DEBUG 0

GLuint bufSampelerProg;
GLuint bufSampelerLoc;
std::string bufSampelerName;

extern UnorderedMap<GLuint, bool> program_map_is_sampler_buffer_emulated;

UnorderedMap<GLuint, SamplerInfo> g_samplerCacheForSamplerBuffer;

namespace {

// The unit gl/texture.cpp parks the emulated buffer texture on. Kept in step with
// MG_TEXTURE_BUFFER_EMULATION_UNIT there and with gl/buffer.cpp's glTexBuffer,
// which borrows the same one.
const GLint kBufferTextureUnit = 15;

// Everything the two program maps have to say about one program, held so the
// sampler list is not copied out of the map on every draw.
//
// It is not a cache of the lookups themselves. Both maps are re-probed on every
// call, because both carry this layer's invalidation points and neither can reach
// a file-local cache: glCreateProgram clears
// program_map_is_sampler_buffer_emulated[program] and erases the sampler entry --
// it exists for that, because GL hands the name of a deleted program straight back
// out -- and glAttachShader sets the flag. Remembering either answer let a recycled
// name keep the previous program's uniform locations, and glUseProgram cannot be
// the invalidation point instead: it is filtered when the name repeats, and a
// recycled name does repeat.
//
// What is kept is the copy below, valid only while the sampler entry it came from
// is still the entry the map holds for this program. Copied rather than pointed at
// across calls because g_samplerCacheForSamplerBuffer is process-wide with no lock,
// so a pointer into it would not survive a rehash or an erase performed on another
// thread.
struct resolved_program_t {
    GLuint program = 0;
    // The entry the copy below was taken from, and the only thing that says the
    // copy still describes this program. A mismatch costs one copy, never a wrong
    // answer; the map's entries are inserted by this function alone, always from
    // the live program, so a hit at the same address is the entry that was copied.
    const SamplerInfo* source = nullptr;
    bool emulated = false;
    GLint locWidth = -1;
    GLint locHeight = -1;
    std::vector<GLint> samplers;
};

// thread_local because gl_state is: two threads with different current contexts
// have different current programs and must not share this.
thread_local resolved_program_t g_resolved_program;

const resolved_program_t& resolve_program(GLuint program) {
    // find() rather than operator[]: this used to insert a default-constructed
    // entry for every program the application ever drew with, on the draw path,
    // and could rehash the map while doing it.
    const auto emu = program_map_is_sampler_buffer_emulated.find(program);
    if (emu == program_map_is_sampler_buffer_emulated.end() || !emu->second) {
        g_resolved_program.program = program;
        g_resolved_program.source = nullptr;
        g_resolved_program.emulated = false;
        g_resolved_program.locWidth = -1;
        g_resolved_program.locHeight = -1;
        g_resolved_program.samplers.clear();
        return g_resolved_program;
    }

    auto it = g_samplerCacheForSamplerBuffer.find(program);
    if (it != g_samplerCacheForSamplerBuffer.end() && g_resolved_program.program == program &&
        g_resolved_program.source == &it->second) {
        return g_resolved_program;
    }

    const SamplerInfo* info = nullptr;
    if (it != g_samplerCacheForSamplerBuffer.end()) {
        info = &it->second;
    } else {
        // Value-initialised: SamplerInfo has no default member initialisers, and
        // the entry is stored even when the program turns out not to carry the
        // emulation uniforms. The old code inserted the entry *before* that check
        // and returned without filling it in, so every later draw with the same
        // program read whatever the allocation happened to hold -- and reprobing
        // was skipped anyway because the key was present.
        SamplerInfo built{};
        built.locWidth = GLES.glGetUniformLocation(program, "u_BufferTexWidth");
        built.locHeight = GLES.glGetUniformLocation(program, "u_BufferTexHeight");
        if (built.locWidth == -1) {
            LOG_W("u_BufferTexWidth uniform not found in program %d", program);
        } else {
            GLint numUniforms = 0;
            GLES.glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &numUniforms);
            LOG_D("Program %d has %d active uniforms", program, numUniforms);

            for (GLint i = 0; i < numUniforms; ++i) {
                const GLsizei bufSize = 256;
                GLchar name[bufSize];
                GLsizei length = 0;
                GLint size = 0;
                GLenum type = 0;
                GLES.glGetActiveUniform(program, i, bufSize, &length, &size, &type, name);

                if (type == GL_SAMPLER_2D || type == GL_INT_SAMPLER_2D) {
                    built.samplers.push_back(GLES.glGetUniformLocation(program, name));
                }
            }
        }
        info = &(g_samplerCacheForSamplerBuffer[program] = std::move(built));
    }

    g_resolved_program.program = program;
    g_resolved_program.source = info;
    g_resolved_program.samplers.clear();

    // A program with no u_BufferTexWidth has nothing to receive, which is what the
    // early return used to say.
    if (info->locWidth == -1) {
        g_resolved_program.emulated = false;
        g_resolved_program.locWidth = -1;
        g_resolved_program.locHeight = -1;
        return g_resolved_program;
    }

    g_resolved_program.emulated = true;
    g_resolved_program.locWidth = info->locWidth;
    g_resolved_program.locHeight = info->locHeight;
    g_resolved_program.samplers = info->samplers;
    return g_resolved_program;
}

} // namespace

void setupBufferTextureUniforms(GLuint program) {
    LOG_D("setupBufferTextureUniforms, program: %d", program);

    const resolved_program_t& info = resolve_program(program);
    if (!info.emulated || info.samplers.empty()) return;

    // The uniform writes stay on the draw path rather than moving to glUseProgram.
    // The size uniforms describe whatever texture is parked on the emulation unit
    // *now*, and an application is entitled to bind its buffer texture after
    // glUseProgram and before the draw -- resolving them at glUseProgram time would
    // describe the previous binding. Only the map lookups above are hoisted, and
    // those depend on the program alone.
    //
    // Every sampler in the program reads that one binding, so it is resolved once
    // for the whole loop instead of once per sampler, as are the size uniforms.
    GLuint texId = 0;
    if (!mg_driver_texture_binding_at_unit(kBufferTextureUnit, GL_TEXTURE_2D, &texId)) {
        // The tracked driver-side binding is not trustworthy right now (FSR1 leaves
        // a texture bound on a unit nothing records), so pay for the round trip.
        // Borrowing the unit has to hand it back: gl/buffer.cpp's glTexBuffer and
        // gl/texture.cpp's glBindTexture both assume the emulation unit is only
        // ever active inside a window that restores it.
        const int prev_unit = mg_driver_active_texture_unit();
        GLES.glActiveTexture(GL_TEXTURE0 + kBufferTextureUnit);
        GLint queried = 0;
        GLES.glGetIntegerv(GL_TEXTURE_BINDING_2D, &queried);
        GLES.glActiveTexture(GL_TEXTURE0 + prev_unit);
        texId = static_cast<GLuint>(queried);
    }
    if (texId == 0) return;

    const TextureObject* texObject = mgGetTexObjectByID(texId);
    // mgGetTexObjectByID answers null for a name this layer has no record of. The
    // dimensions are the whole point of these uniforms, so there is nothing useful
    // to write without it.
    if (!texObject) return;

    bool wrote_sampler = false;
    for (const GLint locSampler : info.samplers) {
        if (locSampler < 0) continue;
        GLES.glUniform1i(locSampler, kBufferTextureUnit);
        wrote_sampler = true;
    }
    if (!wrote_sampler) return;

    GLES.glUniform1i(info.locWidth, texObject->width);
    GLES.glUniform1i(info.locHeight, texObject->height);
}


namespace {

// Flashback/Dear ImGui uses a small RGBA font/UI atlas.  MobileGlues 1.3.5
// rendered it correctly while the 2.x pipeline can end up sampling the same
// underlying RGBA image as black/opaque.  An FBO readback of the texture does
// not apply texture swizzle, so a bad driver-side swizzle is invisible to the
// readback diagnostics even though it changes texture() in the fragment shader.
//
// Query the *real GLES object* immediately before the ImGui draw.  The frontend
// TextureObject shadow is intentionally not consulted here: glTexImage2D resets
// that shadow to RGBA identity, so it can say identity even when a previous
// driver-side parameter call left the real object on another swizzle.
struct imgui_sampling_cache_t {
    GLuint program = 0;
    GLint proj = -1;
    GLint texture = -1;
    bool imgui = false;
};

thread_local imgui_sampling_cache_t g_imgui_sampling_cache;

const imgui_sampling_cache_t& resolve_imgui_sampling(GLuint program) {
    if (g_imgui_sampling_cache.program == program) return g_imgui_sampling_cache;

    g_imgui_sampling_cache = {};
    g_imgui_sampling_cache.program = program;
    if (program == 0) return g_imgui_sampling_cache;

    g_imgui_sampling_cache.proj = GLES.glGetUniformLocation(program, "ProjMtx");
    g_imgui_sampling_cache.texture = GLES.glGetUniformLocation(program, "Texture");
    g_imgui_sampling_cache.imgui = g_imgui_sampling_cache.proj >= 0 && g_imgui_sampling_cache.texture >= 0;
    return g_imgui_sampling_cache;
}

struct imgui_sampler_guard_t {
    GLint unit = -1;
    GLint previous = 0;

    explicit imgui_sampler_guard_t(GLuint program) {
        if (!GLES.glBindSampler || !GLES.glGetUniformiv || !GLES.glGetIntegerv) return;
        const auto& info = resolve_imgui_sampling(program);
        if (!info.imgui) return;

        GLint requested_unit = -1;
        GLES.glGetUniformiv(program, info.texture, &requested_unit);
        if (requested_unit < 0 || requested_unit >= 32) return;

        GLint old_active = GL_TEXTURE0;
        GLES.glGetIntegerv(GL_ACTIVE_TEXTURE, &old_active);
        GLES.glActiveTexture(GL_TEXTURE0 + requested_unit);
        GLint bound_texture = 0;
        GLES.glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound_texture);
        GLES.glGetIntegerv(GL_SAMPLER_BINDING, &previous);
        GLES.glActiveTexture(static_cast<GLenum>(old_active));
        if (bound_texture == 0 || previous == 0) return;

        unit = requested_unit;
        static thread_local int reports = 0;
        if (reports++ < 8) {
            GLint min_filter = -1, mag_filter = -1, wrap_s = -1, compare_mode = -1;
            if (GLES.glGetSamplerParameteriv) {
                GLES.glGetSamplerParameteriv(static_cast<GLuint>(previous), GL_TEXTURE_MIN_FILTER, &min_filter);
                GLES.glGetSamplerParameteriv(static_cast<GLuint>(previous), GL_TEXTURE_MAG_FILTER, &mag_filter);
                GLES.glGetSamplerParameteriv(static_cast<GLuint>(previous), GL_TEXTURE_WRAP_S, &wrap_s);
                GLES.glGetSamplerParameteriv(static_cast<GLuint>(previous), GL_TEXTURE_COMPARE_MODE, &compare_mode);
            }
            LOG_W_FORCE("[MG-IMGUI-SAMPLER] program=%u unit=%d texture=%d previousSampler=%d min=0x%x mag=0x%x wrapS=0x%x compare=0x%x; temporarily unbinding",
                        program, unit, bound_texture, previous, min_filter, mag_filter, wrap_s, compare_mode)
        }
        GLES.glBindSampler(static_cast<GLuint>(unit), 0);
    }

    ~imgui_sampler_guard_t() {
        if (unit >= 0) GLES.glBindSampler(static_cast<GLuint>(unit), static_cast<GLuint>(previous));
    }
    imgui_sampler_guard_t(const imgui_sampler_guard_t&) = delete;
    imgui_sampler_guard_t& operator=(const imgui_sampler_guard_t&) = delete;
};

void restore_imgui_atlas_driver_swizzle(GLuint program) {
    const imgui_sampling_cache_t& info = resolve_imgui_sampling(program);
    if (!info.imgui || GLES.glGetTexParameteriv == nullptr || GLES.glTexParameteri == nullptr) return;

    GLint unit = 0;
    GLES.glGetUniformiv(program, info.texture, &unit);
    if (unit < 0 || unit >= 32) return;

    GLint saved_active = GL_TEXTURE0;
    GLES.glGetIntegerv(GL_ACTIVE_TEXTURE, &saved_active);
    GLES.glActiveTexture(GL_TEXTURE0 + unit);

    GLint texture = 0;
    GLES.glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture);
    if (texture == 0) {
        GLES.glActiveTexture(static_cast<GLenum>(saved_active));
        return;
    }

    GLint width = 0;
    GLint height = 0;
    GLint internal = 0;
    if (GLES.glGetTexLevelParameteriv != nullptr) {
        GLES.glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
        GLES.glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
#ifdef GL_TEXTURE_INTERNAL_FORMAT
        GLES.glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &internal);
#endif
    }

    GLint swizzle_r = GL_RED;
    GLint swizzle_g = GL_GREEN;
    GLint swizzle_b = GL_BLUE;
    GLint swizzle_a = GL_ALPHA;
    GLES.glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, &swizzle_r);
    GLES.glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, &swizzle_g);
    GLES.glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, &swizzle_b);
    GLES.glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, &swizzle_a);

    // This is the Flashback font/UI atlas observed on both the 1.21.9-1.21.11
    // and 26.1.x paths.  Keep the workaround tightly scoped so an ImGui image
    // widget with an intentional custom swizzle is not changed.
    const bool looks_like_flashback_atlas = width == 512 && height == 256 &&
        (internal == GL_RGBA || internal == GL_RGBA8 || internal == 0);
    const bool identity = swizzle_r == GL_RED && swizzle_g == GL_GREEN &&
                          swizzle_b == GL_BLUE && swizzle_a == GL_ALPHA;

    bool repaired = false;
    if (looks_like_flashback_atlas && !identity) {
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_BLUE);
        GLES.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);
        repaired = true;
    }

    static thread_local GLuint last_texture = 0;
    static thread_local GLint last_r = -1, last_g = -1, last_b = -1, last_a = -1;
    static thread_local int logs = 0;
    if (logs < 24 && (texture != static_cast<GLint>(last_texture) ||
                      swizzle_r != last_r || swizzle_g != last_g ||
                      swizzle_b != last_b || swizzle_a != last_a || repaired)) {
        LOG_W_FORCE("[MobileGlues][ImGui REAL swizzle] program=%u unit=%d tex=%d size=%dx%d internal=0x%x driver=%d/%d/%d/%d expected=%d/%d/%d/%d repaired=%d",
                    program, unit, texture, width, height, internal,
                    swizzle_r, swizzle_g, swizzle_b, swizzle_a,
                    GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA, repaired ? 1 : 0)
        ++logs;
        last_texture = static_cast<GLuint>(texture);
        last_r = swizzle_r;
        last_g = swizzle_g;
        last_b = swizzle_b;
        last_a = swizzle_a;
    }

    GLES.glActiveTexture(static_cast<GLenum>(saved_active));
}

} // namespace

// Lightweight ImGui-only probe, automatically active for the first four draws.
// Set MG_IMGUI_GL_PROBE=0 before launch to disable. Driver and frontend state
// are queried separately; the probe does not change rendering state.
static void mg_imgui_probe_draw_state(GLuint program) {
    const auto& info = resolve_imgui_sampling(program);
    if (!info.imgui) return;

    const char* probe = std::getenv("MG_IMGUI_GL_PROBE");
    const bool trace = probe == nullptr || std::strcmp(probe, "0") != 0;
    const char* force = std::getenv("MG_IMGUI_FORCE_BLEND");
    const bool force_blend = force != nullptr && std::strcmp(force, "1") == 0;
    if (!trace && !force_blend) return;

    if (trace) {
        static thread_local int seen = 0;
        if (seen++ < 4) {
            GLint fbo = 0, active_tex = 0, bound_tex = 0, bound_sampler = 0;
            GLint vao = 0, ebo = 0, pbo = 0;
            GLint src_rgb = 0, dst_rgb = 0, src_a = 0, dst_a = 0;
            GLint eq_rgb = 0, eq_a = 0;
            GLboolean color_mask[4] = {0,0,0,0};
            const GLint color_attrib = GLES.glGetAttribLocation(program, "Color");
            GLint color_type = 0, color_norm = 0, color_size = 0, color_stride = 0, color_vbo = 0;
            if (color_attrib >= 0 && GLES.glGetVertexAttribiv != nullptr) {
                GLES.glGetVertexAttribiv(static_cast<GLuint>(color_attrib), GL_VERTEX_ATTRIB_ARRAY_TYPE, &color_type);
                GLES.glGetVertexAttribiv(static_cast<GLuint>(color_attrib), GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &color_norm);
                GLES.glGetVertexAttribiv(static_cast<GLuint>(color_attrib), GL_VERTEX_ATTRIB_ARRAY_SIZE, &color_size);
                GLES.glGetVertexAttribiv(static_cast<GLuint>(color_attrib), GL_VERTEX_ATTRIB_ARRAY_STRIDE, &color_stride);
                GLES.glGetVertexAttribiv(static_cast<GLuint>(color_attrib), GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &color_vbo);
            }
            GLES.glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &fbo);
            GLES.glGetIntegerv(GL_ACTIVE_TEXTURE, &active_tex);
            GLES.glGetIntegerv(GL_TEXTURE_BINDING_2D, &bound_tex);
            GLES.glGetIntegerv(GL_SAMPLER_BINDING, &bound_sampler);
            GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
            GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo);
            GLES.glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &pbo);
            GLES.glGetIntegerv(GL_BLEND_SRC_RGB, &src_rgb);
            GLES.glGetIntegerv(GL_BLEND_DST_RGB, &dst_rgb);
            GLES.glGetIntegerv(GL_BLEND_SRC_ALPHA, &src_a);
            GLES.glGetIntegerv(GL_BLEND_DST_ALPHA, &dst_a);
            GLES.glGetIntegerv(GL_BLEND_EQUATION_RGB, &eq_rgb);
            GLES.glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &eq_a);
            GLES.glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
            LOG_W_FORCE("[MG-IMGUI-PROBE] program=%u fbo=%d activeTex=%d texture=%d sampler=%d vao=%d ebo=%d unpackPBO=%d",
                        program, fbo, active_tex-GL_TEXTURE0, bound_tex, bound_sampler, vao, ebo, pbo)
            LOG_W_FORCE("[MG-IMGUI-PROBE] blend(driver/frontend)=%d/%d depth=%d scissor=%d srcRGB=%d dstRGB=%d srcA=%d dstA=%d eqRGB=%d eqA=%d rgbaMask=%d%d%d%d",
                        GLES.glIsEnabled(GL_BLEND) ? 1 : 0, mg_enable_get(GL_BLEND,0) ? 1 : 0,
                        GLES.glIsEnabled(GL_DEPTH_TEST) ? 1 : 0, GLES.glIsEnabled(GL_SCISSOR_TEST) ? 1 : 0,
                        src_rgb, dst_rgb, src_a, dst_a, eq_rgb, eq_a,
                        color_mask[0],color_mask[1],color_mask[2],color_mask[3])
            LOG_W_FORCE("[MG-IMGUI-PROBE] colorAttrib=%d type=0x%x normalized=%d size=%d stride=%d vbo=%d (expected normalized=1 for packed UBYTE color)",
                        color_attrib, color_type, color_norm, color_size, color_stride, color_vbo)
        }
    }

    if (force_blend) {
        // Diagnostic only. ImGui's Java renderer already requests precisely this
        // state; forcing it through GLES tests whether MG suppressed that request.
        GLES.glEnable(GL_BLEND);
        GLES.glBlendEquation(GL_FUNC_ADD);
        GLES.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }
}

void prepareForDraw() {
    LOG_D("prepareForDraw...")
    if (hardware->emulate_texture_buffer) {
        setupBufferTextureUniforms(gl_state->current_program);
    }
    restore_imgui_atlas_driver_swizzle(gl_state->current_program);
    mg_imgui_probe_draw_state(gl_state->current_program);
}

void glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei primcount) {

    static thread_local int dh262_dei = 0;
    if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_dei++ < 80) {
        GLint vao = 0, ebo = 0;
        GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo);
        LOG_W_FORCE("[DH26.2 DRAW] glDrawElementsInstanced mode=0x%x count=%d type=0x%x indices=%p instances=%d vao=%d ebo=%d",
                    mode, count, type, indices, primcount, vao, ebo)
    }
    LOG()
    LOG_D("glDrawElementsInstanced, mode: %d, count: %d, type: %d, indices: %p, primcount: %d", mode, count, type,
          indices, primcount)
    prepareForDraw();
    if (mg_restart_needs_rewrite(type) && mg_draw_elements_restart(mode, count, type, indices, 0, primcount)) return;
    const bool restart_fixed = mg_restart_needs_driver_fixed(type);
    if (restart_fixed) GLES.glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    GLES.glDrawElementsInstanced(mode, count, type, indices, primcount);
    if (restart_fixed) GLES.glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    CHECK_GL_ERROR
}

void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {

    static thread_local int dh262_de = 0;
    if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_de++ < 80) {
        GLint vao = 0, ebo = 0;
        GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo);
        LOG_W_FORCE("[DH26.2 DRAW] glDrawElements mode=0x%x count=%d type=0x%x indices=%p vao=%d ebo=%d",
                    mode, count, type, indices, vao, ebo)
    }
    LOG()
    LOG_D("glDrawElements, mode: %d, count: %d, type: %d, indices: %p", mode, count, type, indices)
    prepareForDraw();
    imgui_sampler_guard_t imgui_sampler_guard(gl_state->current_program);
    if (mg_restart_needs_rewrite(type) && mg_draw_elements_restart(mode, count, type, indices, 0, -1)) return;
    const bool restart_fixed = mg_restart_needs_driver_fixed(type);
    if (restart_fixed) GLES.glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    GLES.glDrawElements(mode, count, type, indices);
    if (restart_fixed) GLES.glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    CHECK_GL_ERROR
}

void glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access,
                        GLenum format) {
    LOG()
    LOG_D("glBindImageTexture, unit: %d, texture: %d, level: %d, layered: %d, layer: %d, access: %d, format: %d", unit,
          texture, level, layered, layer, access, format)
    GLES.glBindImageTexture(unit, texture, level, layered, layer, access, format);
    CHECK_GL_ERROR
}

void glUniform1i(GLint location, GLint v0) {
    LOG()
    LOG_D("glUniform1i, location: %d, v0: %d", location, v0)
    GLES.glUniform1i(location, v0);
    CHECK_GL_ERROR
}

void glDispatchCompute(GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z) {
    LOG()
    LOG_D("glDispatchCompute, num_groups_x: %d, num_groups_y: %d, num_groups_z: %d", num_groups_x, num_groups_y,
          num_groups_z)
    GLES.glDispatchCompute(num_groups_x, num_groups_y, num_groups_z);
    CHECK_GL_ERROR
}

void glMemoryBarrier(GLbitfield barriers) {
    LOG()
    LOG_D("glMemoryBarrier, barriers: %d", barriers)
    GLES.glMemoryBarrier(barriers);
    CHECK_GL_ERROR
}

namespace {

// Scratch index buffer for the base-vertex emulation below, and the context that
// owns it. Modeled on gl/restart.cpp's g_restart_ibo, including the invalidation:
// thread_local because g_current_ctx is, so two threads with different current
// contexts keep their own name instead of trading one back and forth.
thread_local GLuint g_basevertex_ibo = 0;
thread_local unsigned long long g_basevertex_owner_ctx_id = 0;

// Drop the cached name when the current context is not the one that created it.
//
// Deliberately no glDeleteBuffers: if the owning context is gone the buffer went
// with it, and if it is merely not current then this name refers to a buffer
// belonging to whichever context *is* current -- the glBufferData below would
// overwrite that buffer's contents.
void basevertex_check_context() {
    const unsigned long long cur = g_current_ctx ? g_current_ctx->id : 0;
    if (cur == g_basevertex_owner_ctx_id) return;
    g_basevertex_ibo = 0;
    g_basevertex_owner_ctx_id = cur;
}

// Staging for the rebased index stream. Elements are GLuint so the storage is
// always aligned for the widest index type it has to hold; the length is in whole
// GLuints, rounded up. Grown and never shrunk, so a steady stream of draws
// allocates nothing -- this used to be a malloc and a free per call.
thread_local std::vector<GLuint> g_basevertex_staging;

void* basevertex_staging(size_t bytes) {
    // Grown only. A plain resize() to the exact length shrinks after a small draw
    // and then value-initialises the difference on the next large one -- a memset
    // of the whole tail that the caller's memcpy overwrites immediately. Only the
    // first `bytes` bytes are ever read, so what is past them is out of range in
    // the same way it is in gl/multidraw.cpp and gl/restart.cpp.
    const size_t need = (bytes + sizeof(GLuint) - 1) / sizeof(GLuint);
    if (g_basevertex_staging.size() < need) g_basevertex_staging.resize(need);
    return g_basevertex_staging.data();
}

} // namespace

void glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLint basevertex) {

    static thread_local int dh262_debv = 0;
    if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_debv++ < 80) {
        GLint vao = 0, ebo = 0;
        GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo);
        LOG_W_FORCE("[DH26.2 DRAW] glDrawElementsBaseVertex mode=0x%x count=%d type=0x%x indices=%p base=%d vao=%d ebo=%d",
                    mode, count, type, indices, basevertex, vao, ebo)
    }
    LOG()
    LOG_D("glDrawElementsBaseVertex, mode: %d, count: %d, type: %d, indices: %p, basevertex: %d", mode, count, type,
          indices, basevertex);
    prepareForDraw();
    imgui_sampler_guard_t imgui_sampler_guard(gl_state->current_program);
    // The rewrite applies the base vertex itself, so it covers both the emulated
    // and the driver-supported branch below.
    if (mg_restart_needs_rewrite(type) && mg_draw_elements_restart(mode, count, type, indices, basevertex, -1)) return;
    const bool restart_fixed = mg_restart_needs_driver_fixed(type);
    if (restart_fixed) GLES.glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    struct RestartGuard {
        bool on;
        ~RestartGuard() {
            if (on) GLES.glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
        }
    } restart_guard{restart_fixed};
    if (hardware->es_version < 320 && !g_gles_caps.GL_EXT_draw_elements_base_vertex &&
        !g_gles_caps.GL_OES_draw_elements_base_vertex) {
        // TODO: use indirect drawing for GLES 3.1
        LOG_D("Emulating glDrawElementsBaseVertex")
        if (basevertex == 0) {
            GLES.glDrawElements(mode, count, type, indices);
            return;
        }
        if (count <= 0) return;

        size_t indexSize;
        switch (type) {
        case GL_UNSIGNED_INT:
            indexSize = sizeof(GLuint);
            break;
        case GL_UNSIGNED_SHORT:
            indexSize = sizeof(GLushort);
            break;
        case GL_UNSIGNED_BYTE:
            indexSize = sizeof(GLubyte);
            break;
        default:
            return;
        }

        const size_t bytes = static_cast<size_t>(count) * indexSize;

        // The tracked binding rather than a driver round trip. It is the driver's
        // name, so it goes straight back to GLES.glBindBuffer, and it is asked for
        // before the temporary bind below, which is the only window in which this
        // function makes the two disagree. gl/gl.cpp's depth-clear triangle is the
        // one path in the layer that leaves the driver on a vertex array the
        // tracked state does not follow, and the element array binding is vertex
        // array state; see the note on the accessor in gl/buffer.cpp.
        const GLuint prevElementBuffer = mg_driver_bound_buffer(GL_ELEMENT_ARRAY_BUFFER);

        void* tempIndices = basevertex_staging(bytes);

        if (prevElementBuffer != 0) {
            // Redundant on paper -- that buffer is already bound -- but it is what
            // guarantees the map below reads the buffer this call resolved, the
            // same way gl/restart.cpp and gl/multidraw.cpp do it.
            GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);
            // Read-only, and it has to stay a map: gl/buffer.cpp tracks a buffer's
            // size but never its contents, so there is no shadow copy of the index
            // data to rebase from.
            void* srcData =
                GLES.glMapBufferRange(GL_ELEMENT_ARRAY_BUFFER,
                                      static_cast<GLintptr>(reinterpret_cast<uintptr_t>(indices)),
                                      static_cast<GLsizeiptr>(bytes), GL_MAP_READ_BIT);
            if (!srcData) {
                // An immutable or persistently mapped index buffer cannot be read
                // back, and there is no driver base vertex on this path to fall
                // back to. Drop the draw rather than place the geometry at the
                // wrong vertices.
                return;
            }
            memcpy(tempIndices, srcData, bytes);
            GLES.glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
        } else {
            if (!indices) return;
            memcpy(tempIndices, indices, bytes);
        }

        switch (type) {
        case GL_UNSIGNED_INT:
            for (GLsizei j = 0; j < count; ++j) {
                ((GLuint*)tempIndices)[j] += basevertex;
            }
            break;
        case GL_UNSIGNED_SHORT:
            for (GLsizei j = 0; j < count; ++j) {
                ((GLushort*)tempIndices)[j] += basevertex;
            }
            break;
        case GL_UNSIGNED_BYTE:
            for (GLsizei j = 0; j < count; ++j) {
                ((GLubyte*)tempIndices)[j] += basevertex;
            }
            break;
        }

        // One persistent scratch buffer instead of a glGenBuffers/glDeleteBuffers
        // pair per draw call.
        basevertex_check_context();
        if (!g_basevertex_ibo) GLES.glGenBuffers(1, &g_basevertex_ibo);
        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_basevertex_ibo);
        GLES.glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(bytes), tempIndices, GL_STREAM_DRAW);

        GLES.glDrawElements(mode, count, type, nullptr);

        GLES.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, prevElementBuffer);

        CHECK_GL_ERROR
    } else {
        GLES.glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
    }
    CHECK_GL_ERROR
}

#define DR_WARN_ONCE(...)                                                                                              \
    do {                                                                                                               \
        static bool mg_dr_warned = false;                                                                              \
        if (!mg_dr_warned) {                                                                                           \
            mg_dr_warned = true;                                                                                       \
            LOG_W_FORCE(__VA_ARGS__)                                                                                   \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// The rest of the indexed draw family
//
// These were pass-throughs in gl/gl_native.cpp, so GL_PRIMITIVE_RESTART with a
// custom index went straight to a driver that has no such feature and every
// restart in the batch was drawn as ordinary geometry -- strips joined end to
// end. They are indexed draws like the three above and owe the same treatment:
// rewrite the stream when the chosen value is not the fixed one, and otherwise
// switch the driver's fixed-index restart on for the duration.
// ---------------------------------------------------------------------------

// Brackets a draw with GLES' fixed-index restart. Scoped so an early return
// cannot leave it enabled behind the application's back.
namespace {
struct restart_guard_t {
    bool on;
    explicit restart_guard_t(GLenum type) : on(mg_restart_needs_driver_fixed(type)) {
        if (on) GLES.glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    }
    ~restart_guard_t() {
        if (on) GLES.glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    }
    restart_guard_t(const restart_guard_t&) = delete;
    restart_guard_t& operator=(const restart_guard_t&) = delete;
};
} // namespace

void glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices) {
    LOG()
    LOG_D("glDrawRangeElements, mode: %d, start: %u, end: %u, count: %d, type: %d", mode, start, end, count, type)
    prepareForDraw();
    // The rewritten stream is 32-bit with 0xFFFFFFFF sentinels, so start/end no
    // longer describe it. They are only a promise about the index range, and
    // dropping the promise is allowed; drawing the wrong primitives is not.
    if (mg_restart_needs_rewrite(type) && mg_draw_elements_restart(mode, count, type, indices, 0, -1)) return;
    restart_guard_t guard(type);
    GLES.glDrawRangeElements(mode, start, end, count, type, indices);
    CHECK_GL_ERROR
}

void glDrawRangeElementsBaseVertex(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type,
                                   const void* indices, GLint basevertex) {
    LOG()
    LOG_D("glDrawRangeElementsBaseVertex, mode: %d, count: %d, type: %d, basevertex: %d", mode, count, type, basevertex)
    prepareForDraw();
    if (mg_restart_needs_rewrite(type) && mg_draw_elements_restart(mode, count, type, indices, basevertex, -1)) return;
    restart_guard_t guard(type);
    if (GLES.glDrawRangeElementsBaseVertex) {
        GLES.glDrawRangeElementsBaseVertex(mode, start, end, count, type, indices, basevertex);
    } else {
        // glDrawElementsBaseVertex above already emulates the base vertex when
        // the driver cannot; the range is the only thing lost.
        glDrawElementsBaseVertex(mode, count, type, indices, basevertex);
    }
    CHECK_GL_ERROR
}


namespace {

bool dh262_is_lod_packed_signature() {
    GLint en0 = 0, size0 = 0, type0 = 0, int0 = 0;
    GLint en5 = 0, size5 = 0, type5 = 0, int5 = 0;
    GLES.glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &en0);
    GLES.glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_SIZE, &size0);
    GLES.glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_TYPE, &type0);
#ifdef GL_VERTEX_ATTRIB_ARRAY_INTEGER
    GLES.glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_INTEGER, &int0);
#endif
    GLES.glGetVertexAttribiv(5, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &en5);
    GLES.glGetVertexAttribiv(5, GL_VERTEX_ATTRIB_ARRAY_SIZE, &size5);
    GLES.glGetVertexAttribiv(5, GL_VERTEX_ATTRIB_ARRAY_TYPE, &type5);
#ifdef GL_VERTEX_ATTRIB_ARRAY_INTEGER
    GLES.glGetVertexAttribiv(5, GL_VERTEX_ATTRIB_ARRAY_INTEGER, &int5);
#endif
    return en0 && int0 && size0 == 3 && type0 == GL_UNSIGNED_SHORT &&
           en5 && int5 && size5 == 1 && type5 == GL_UNSIGNED_SHORT;
}

struct dh262_lod_depth_compat_guard_t {
    bool changed = false;
    GLint previous = GL_LESS;

    explicit dh262_lod_depth_compat_guard_t(bool enable) {
        if (!enable) return;
        if (GLES.glIsEnabled(GL_DEPTH_TEST) != GL_TRUE) return;
        GLES.glGetIntegerv(GL_DEPTH_FUNC, &previous);
        if (previous == GL_GREATER) {
            GLES.glDepthFunc(GL_ALWAYS);
            changed = true;
            static thread_local int dh262_depth_fix_log = 0;
            if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_depth_fix_log++ < 64) {
                LOG_W_FORCE("[DH26.2 DEPTH FIX] packed LOD draw: GL_GREATER -> GL_ALWAYS for this draw")
            }
        }
    }

    ~dh262_lod_depth_compat_guard_t() {
        if (changed) GLES.glDepthFunc(static_cast<GLenum>(previous));
    }

    dh262_lod_depth_compat_guard_t(const dh262_lod_depth_compat_guard_t&) = delete;
    dh262_lod_depth_compat_guard_t& operator=(const dh262_lod_depth_compat_guard_t&) = delete;
};

bool dh262_mark_program_for_interface_dump(GLuint program) {
    static thread_local GLuint seen[32] = {};
    static thread_local int seen_count = 0;
    for (int i = 0; i < seen_count; ++i) {
        if (seen[i] == program) return false;
    }
    if (seen_count < 32) seen[seen_count++] = program;
    return true;
}

void dh262_dump_program_interface(GLuint program) {
    GLint linked = 0, attrib_count = 0, uniform_count = 0, block_count = 0;
    GLES.glGetProgramiv(program, GL_LINK_STATUS, &linked);
    GLES.glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &attrib_count);
    GLES.glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &uniform_count);
    GLES.glGetProgramiv(program, GL_ACTIVE_UNIFORM_BLOCKS, &block_count);
    LOG_W_FORCE("[DH26.2 LOD IFACE] program=%u linked=%d activeAttribs=%d activeUniforms=%d uniformBlocks=%d",
                program, linked, attrib_count, uniform_count, block_count)

    const GLint attr_limit = attrib_count < 32 ? attrib_count : 32;
    for (GLint i = 0; i < attr_limit; ++i) {
        GLchar name[256] = {};
        GLsizei length = 0;
        GLint size = 0;
        GLenum type = 0;
        GLES.glGetActiveAttrib(program, static_cast<GLuint>(i), 255, &length, &size, &type, name);
        const GLint location = GLES.glGetAttribLocation(program, name);
        LOG_W_FORCE("[DH26.2 LOD IFACE] attribIndex=%d name=%s location=%d size=%d shaderType=0x%x",
                    i, name, location, size, type)
    }

    const GLint uniform_limit = uniform_count < 64 ? uniform_count : 64;
    for (GLint i = 0; i < uniform_limit; ++i) {
        GLchar name[256] = {};
        GLsizei length = 0;
        GLint size = 0;
        GLenum type = 0;
        GLES.glGetActiveUniform(program, static_cast<GLuint>(i), 255, &length, &size, &type, name);
        const GLint location = GLES.glGetUniformLocation(program, name);
        LOG_W_FORCE("[DH26.2 LOD IFACE] uniformIndex=%d name=%s location=%d size=%d type=0x%x",
                    i, name, location, size, type)
        if (location >= 0 && type == GL_FLOAT_MAT4) {
            GLfloat m[16] = {};
            GLES.glGetUniformfv(program, location, m);
            LOG_W_FORCE("[DH26.2 LOD MAT4] program=%u name=%s loc=%d rows=[%.4f %.4f %.4f %.4f] [%.4f %.4f %.4f %.4f] [%.4f %.4f %.4f %.4f] [%.4f %.4f %.4f %.4f]",
                        program, name, location,
                        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                        m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15])
        }
    }

    const GLint block_limit = block_count < 32 ? block_count : 32;
    for (GLint i = 0; i < block_limit; ++i) {
        GLchar name[256] = {};
        GLsizei length = 0;
        GLint binding = 0, data_size = 0, real_ubo = 0;
        GLint64 start = 0, bound_size = 0;
        GLES.glGetActiveUniformBlockName(program, static_cast<GLuint>(i), 255, &length, name);
        GLES.glGetActiveUniformBlockiv(program, static_cast<GLuint>(i), GL_UNIFORM_BLOCK_BINDING, &binding);
        GLES.glGetActiveUniformBlockiv(program, static_cast<GLuint>(i), GL_UNIFORM_BLOCK_DATA_SIZE, &data_size);
        GLES.glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, static_cast<GLuint>(binding), &real_ubo);
        if (GLES.glGetInteger64i_v != nullptr) {
            GLES.glGetInteger64i_v(GL_UNIFORM_BUFFER_START, static_cast<GLuint>(binding), &start);
            GLES.glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, static_cast<GLuint>(binding), &bound_size);
        }
        LOG_W_FORCE("[DH26.2 LOD UBO] program=%u block=%d name=%s binding=%d dataSize=%d realUbo=%d start=%lld boundSize=%lld",
                    program, i, name, binding, data_size, real_ubo,
                    static_cast<long long>(start), static_cast<long long>(bound_size))
    }
}

void dh262_dump_lod_render_state(GLuint program, GLint vao, GLint ebo, GLsizei count, GLenum type, GLint basevertex) {
    static thread_local int state_logs = 0;
    if (state_logs++ >= 120) return;

    GLint fbo = 0, viewport[4] = {}, scissor[4] = {};
    GLint depth_func = 0, cull_mode = 0, front_face = 0;
    GLint blend_src_rgb = 0, blend_dst_rgb = 0, blend_eq_rgb = 0;
    GLint binding0_buffer = 0, binding0_stride = 0;
    GLint64 binding0_offset = 0;
    GLboolean depth_mask = GL_FALSE, color_mask[4] = {};

    GLES.glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);
    GLES.glGetIntegerv(GL_VIEWPORT, viewport);
    GLES.glGetIntegerv(GL_SCISSOR_BOX, scissor);
    GLES.glGetIntegerv(GL_DEPTH_FUNC, &depth_func);
    GLES.glGetIntegerv(GL_CULL_FACE_MODE, &cull_mode);
    GLES.glGetIntegerv(GL_FRONT_FACE, &front_face);
    GLES.glGetIntegerv(GL_BLEND_SRC_RGB, &blend_src_rgb);
    GLES.glGetIntegerv(GL_BLEND_DST_RGB, &blend_dst_rgb);
    GLES.glGetIntegerv(GL_BLEND_EQUATION_RGB, &blend_eq_rgb);
    GLES.glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask);
    GLES.glGetBooleanv(GL_COLOR_WRITEMASK, color_mask);
#ifdef GL_VERTEX_BINDING_BUFFER
    GLES.glGetIntegeri_v(GL_VERTEX_BINDING_BUFFER, 0, &binding0_buffer);
    GLES.glGetIntegeri_v(GL_VERTEX_BINDING_STRIDE, 0, &binding0_stride);
    if (GLES.glGetInteger64i_v != nullptr) {
        GLES.glGetInteger64i_v(GL_VERTEX_BINDING_OFFSET, 0, &binding0_offset);
    }
#endif

    const GLboolean depth_test = GLES.glIsEnabled(GL_DEPTH_TEST);
    const GLboolean cull_face = GLES.glIsEnabled(GL_CULL_FACE);
    const GLboolean blend = GLES.glIsEnabled(GL_BLEND);
    const GLboolean scissor_test = GLES.glIsEnabled(GL_SCISSOR_TEST);
    const GLenum fbo_status = GLES.glCheckFramebufferStatus(GL_FRAMEBUFFER);

    GLint ubo_real[4] = {};
    GLint64 ubo_start[4] = {}, ubo_size[4] = {};
    for (GLuint b = 0; b < 4; ++b) {
        GLES.glGetIntegeri_v(GL_UNIFORM_BUFFER_BINDING, b, &ubo_real[b]);
        if (GLES.glGetInteger64i_v != nullptr) {
            GLES.glGetInteger64i_v(GL_UNIFORM_BUFFER_START, b, &ubo_start[b]);
            GLES.glGetInteger64i_v(GL_UNIFORM_BUFFER_SIZE, b, &ubo_size[b]);
        }
    }

    LOG_W_FORCE("[DH26.2 LOD STATE] program=%u vao=%d ebo=%d count=%d type=0x%x base=%d fbo=%d fboStatus=0x%x viewport=%d,%d,%d,%d",
                program, vao, ebo, count, type, basevertex, fbo, fbo_status,
                viewport[0], viewport[1], viewport[2], viewport[3])
    LOG_W_FORCE("[DH26.2 LOD STATE] depth=%d func=0x%x depthMask=%d cull=%d cullMode=0x%x frontFace=0x%x blend=%d blendSrc=0x%x blendDst=0x%x blendEq=0x%x colorMask=%d%d%d%d",
                depth_test ? 1 : 0, depth_func, depth_mask ? 1 : 0,
                cull_face ? 1 : 0, cull_mode, front_face,
                blend ? 1 : 0, blend_src_rgb, blend_dst_rgb, blend_eq_rgb,
                color_mask[0] ? 1 : 0, color_mask[1] ? 1 : 0,
                color_mask[2] ? 1 : 0, color_mask[3] ? 1 : 0)
    LOG_W_FORCE("[DH26.2 LOD STATE] scissor=%d box=%d,%d,%d,%d binding0Buffer=%d binding0Offset=%lld binding0Stride=%d",
                scissor_test ? 1 : 0, scissor[0], scissor[1], scissor[2], scissor[3],
                binding0_buffer, static_cast<long long>(binding0_offset), binding0_stride)
    LOG_W_FORCE("[DH26.2 LOD UBO STATE] b0=%d/%lld/%lld b1=%d/%lld/%lld b2=%d/%lld/%lld b3=%d/%lld/%lld",
                ubo_real[0], static_cast<long long>(ubo_start[0]), static_cast<long long>(ubo_size[0]),
                ubo_real[1], static_cast<long long>(ubo_start[1]), static_cast<long long>(ubo_size[1]),
                ubo_real[2], static_cast<long long>(ubo_start[2]), static_cast<long long>(ubo_size[2]),
                ubo_real[3], static_cast<long long>(ubo_start[3]), static_cast<long long>(ubo_size[3]))
}

} // namespace

void glDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices,
                                       GLsizei instancecount, GLint basevertex) {

    static thread_local int dh262_vao_snap_count = 0;
    GLint dh262_snapshot_ebo = 0;
    if (DROIDBRIDGE_GL_DIAGNOSTICS) {
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &dh262_snapshot_ebo);
    }
    if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_snapshot_ebo >= 100 && dh262_is_lod_packed_signature() && dh262_vao_snap_count++ < 400) {
        GLint vao = 0, ebo = 0, program = 0;
        GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo);
        GLES.glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        LOG_W_FORCE("[DH26.2 VAO SNAP] draw vao=%d ebo=%d program=%d count=%d type=0x%x indices=%p instances=%d base=%d",
                    vao, ebo, program, count, type, indices, instancecount, basevertex)

        for (GLuint a = 0; a < 8; ++a) {
            GLint enabled = 0, size = 0, type_i = 0, normalized = 0, integer_i = 0, divisor = 0;
            GLES.glGetVertexAttribiv(a, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
            GLES.glGetVertexAttribiv(a, GL_VERTEX_ATTRIB_ARRAY_SIZE, &size);
            GLES.glGetVertexAttribiv(a, GL_VERTEX_ATTRIB_ARRAY_TYPE, &type_i);
            GLES.glGetVertexAttribiv(a, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED, &normalized);
#ifdef GL_VERTEX_ATTRIB_ARRAY_INTEGER
            GLES.glGetVertexAttribiv(a, GL_VERTEX_ATTRIB_ARRAY_INTEGER, &integer_i);
#endif
#ifdef GL_VERTEX_ATTRIB_ARRAY_DIVISOR
            GLES.glGetVertexAttribiv(a, GL_VERTEX_ATTRIB_ARRAY_DIVISOR, &divisor);
#endif
            if (enabled || a < 4) {
                LOG_W_FORCE("[DH26.2 VAO SNAP] attr=%u en=%d size=%d type=0x%x norm=%d int=%d divisor=%d",
                            a, enabled, size, type_i, normalized, integer_i, divisor)
            }
        }
    }


    static thread_local int dh262_deibv = 0;
    if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_deibv++ < 80) {
        GLint vao = 0, ebo = 0;
        GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &ebo);
        LOG_W_FORCE("[DH26.2 DRAW] glDrawElementsInstancedBaseVertex mode=0x%x count=%d type=0x%x indices=%p instances=%d base=%d vao=%d ebo=%d",
                    mode, count, type, indices, instancecount, basevertex, vao, ebo)
    }
    LOG()
    LOG_D("glDrawElementsInstancedBaseVertex, mode: %d, count: %d, type: %d, instancecount: %d, basevertex: %d", mode,
          count, type, instancecount, basevertex)
    prepareForDraw();
    // Program-specific DH 26.2 LOD diagnostics. The packed 16-byte format
    // (ushort3, ushort1, rgba8, ubyte, ubyte, ushort) is distinctive enough to
    // avoid consuming the log budget on ordinary Minecraft world/UI draws.
    if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_is_lod_packed_signature()) {
        GLint dh262_lod_program = 0, dh262_lod_vao = 0, dh262_lod_ebo = 0;
        GLES.glGetIntegerv(GL_CURRENT_PROGRAM, &dh262_lod_program);
        GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &dh262_lod_vao);
        GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &dh262_lod_ebo);
        if (dh262_mark_program_for_interface_dump(static_cast<GLuint>(dh262_lod_program))) {
            dh262_dump_program_interface(static_cast<GLuint>(dh262_lod_program));
        }
        dh262_dump_lod_render_state(static_cast<GLuint>(dh262_lod_program), dh262_lod_vao,
                                    dh262_lod_ebo, count, type, basevertex);
    }
    // Do not rewrite DH's requested depth function. Older diagnostic builds
    // changed GL_GREATER to GL_ALWAYS here, which leaks test behavior into normal rendering.
    dh262_lod_depth_compat_guard_t dh262_depth_guard(false);

    if (mg_restart_needs_rewrite(type) &&
        mg_draw_elements_restart(mode, count, type, indices, basevertex, instancecount))
        return;
    restart_guard_t guard(type);
    if (GLES.glDrawElementsInstancedBaseVertex) {
        GLES.glDrawElementsInstancedBaseVertex(mode, count, type, indices, instancecount, basevertex);
        GLint dh262_late_ebo = 0;
        if (DROIDBRIDGE_GL_DIAGNOSTICS) {
            GLES.glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &dh262_late_ebo);
        }
        if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_late_ebo >= 100 && dh262_is_lod_packed_signature()) {
            static thread_local int dh262_late_draw_log = 0;
            if (DROIDBRIDGE_GL_DIAGNOSTICS && dh262_late_draw_log++ < 2000) {
                GLint dh262_late_vao = 0, dh262_late_program = 0;
                GLES.glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &dh262_late_vao);
                GLES.glGetIntegerv(GL_CURRENT_PROGRAM, &dh262_late_program);
                GLenum err = GLES.glGetError();
                LOG_W_FORCE("[DH26.2 LOD DRAW] vao=%d eboReal=%d program=%d count=%d type=0x%x indices=%p instances=%d base=%d glesError=0x%x",
                            dh262_late_vao, dh262_late_ebo, dh262_late_program,
                            count, type, indices, instancecount, basevertex, err)
            }
        }
    } else if (basevertex == 0) {
        GLES.glDrawElementsInstanced(mode, count, type, indices, instancecount);
    } else {
        DR_WARN_ONCE("glDrawElementsInstancedBaseVertex: no base vertex support on this context, drawing without it");
        GLES.glDrawElementsInstanced(mode, count, type, indices, instancecount);
    }
    CHECK_GL_ERROR
}

// ---------------------------------------------------------------------------
// The base instance family (GL 4.2 / ARB_base_instance)
//
// GLES has no base instance in core, and no extension for it on the drivers
// this layer targets, so these three were stubs in gl/gl_stub.cpp: called, they
// drew nothing at all. That is the worst of the available options -- a mesh that
// silently never appears is harder to diagnose than one in the wrong place, and
// baseinstance is 0 in the overwhelming majority of calls, where these commands
// are exactly the ones GLES already implements.
//
// So they forward, and a non-zero base instance is reported once and then
// ignored. The instanced attribute fetch then starts at element 0 instead of
// baseinstance, which is wrong for that case only, and stays visible.
// ---------------------------------------------------------------------------

void glDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount,
                                       GLuint baseinstance) {
    LOG()
    LOG_D("glDrawArraysInstancedBaseInstance, mode: %d, first: %d, count: %d, instancecount: %d, baseinstance: %u",
          mode, first, count, instancecount, baseinstance)
    if (baseinstance != 0) {
        DR_WARN_ONCE("glDrawArraysInstancedBaseInstance: baseinstance %u ignored, GLES has no base instance",
                     baseinstance);
    }
    prepareForDraw();
    GLES.glDrawArraysInstanced(mode, first, count, instancecount);
    CHECK_GL_ERROR
}

void glDrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices,
                                         GLsizei instancecount, GLuint baseinstance) {
    LOG()
    LOG_D("glDrawElementsInstancedBaseInstance, mode: %d, count: %d, type: %d, instancecount: %d, baseinstance: %u",
          mode, count, type, instancecount, baseinstance)
    if (baseinstance != 0) {
        DR_WARN_ONCE("glDrawElementsInstancedBaseInstance: baseinstance %u ignored, GLES has no base instance",
                     baseinstance);
    }
    glDrawElementsInstanced(mode, count, type, indices, instancecount);
}

void glDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices,
                                                   GLsizei instancecount, GLint basevertex, GLuint baseinstance) {
    LOG()
    LOG_D("glDrawElementsInstancedBaseVertexBaseInstance, mode: %d, count: %d, basevertex: %d, baseinstance: %u", mode,
          count, basevertex, baseinstance)
    if (baseinstance != 0) {
        DR_WARN_ONCE(
            "glDrawElementsInstancedBaseVertexBaseInstance: baseinstance %u ignored, GLES has no base instance",
            baseinstance);
    }
    glDrawElementsInstancedBaseVertex(mode, count, type, indices, instancecount, basevertex);
}
