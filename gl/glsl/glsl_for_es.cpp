// MobileGlues - gl/glsl/glsl_for_es.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v2.1:
//   https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt
// SPDX-License-Identifier: LGPL-2.1-only
// End of Source File Header
#include "glsl_for_es.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Include/Types.h>
#include <glslang/Public/ShaderLang.h>
#include <spirv_cross/spirv_cross_c.h>
#include <iostream>
#include <fstream>
#include "../log.h"
#include "glslang/SPIRV/GlslangToSpv.h"
#include <string>
#include <regex>
#include <strstream>
#include <algorithm>
#include <sstream>
#include "cache.h"
#include "../../version.h"

#define DEBUG 0

static TBuiltInResource InitResources() {
    TBuiltInResource Resources{};

    Resources.maxLights = 32;
    Resources.maxClipPlanes = 6;
    Resources.maxTextureUnits = 32;
    Resources.maxTextureCoords = 32;
    Resources.maxVertexAttribs = 64;
    Resources.maxVertexUniformComponents = 4096;
    Resources.maxVaryingFloats = 64;
    Resources.maxVertexTextureImageUnits = 32;
    Resources.maxCombinedTextureImageUnits = 80;
    Resources.maxTextureImageUnits = 32;
    Resources.maxFragmentUniformComponents = 4096;
    Resources.maxDrawBuffers = 32;
    Resources.maxVertexUniformVectors = 128;
    Resources.maxVaryingVectors = 8;
    Resources.maxFragmentUniformVectors = 16;
    Resources.maxVertexOutputVectors = 16;
    Resources.maxFragmentInputVectors = 15;
    Resources.minProgramTexelOffset = -8;
    Resources.maxProgramTexelOffset = 7;
    Resources.maxClipDistances = 8;
    Resources.maxComputeWorkGroupCountX = 65535;
    Resources.maxComputeWorkGroupCountY = 65535;
    Resources.maxComputeWorkGroupCountZ = 65535;
    Resources.maxComputeWorkGroupSizeX = 1024;
    Resources.maxComputeWorkGroupSizeY = 1024;
    Resources.maxComputeWorkGroupSizeZ = 64;
    Resources.maxComputeUniformComponents = 1024;
    Resources.maxComputeTextureImageUnits = 16;
    Resources.maxComputeImageUniforms = 8;
    Resources.maxComputeAtomicCounters = 8;
    Resources.maxComputeAtomicCounterBuffers = 1;
    Resources.maxVaryingComponents = 60;
    Resources.maxVertexOutputComponents = 64;
    Resources.maxGeometryInputComponents = 64;
    Resources.maxGeometryOutputComponents = 128;
    Resources.maxFragmentInputComponents = 128;
    Resources.maxImageUnits = 8;
    Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
    Resources.maxCombinedShaderOutputResources = 8;
    Resources.maxImageSamples = 0;
    Resources.maxVertexImageUniforms = 0;
    Resources.maxTessControlImageUniforms = 0;
    Resources.maxTessEvaluationImageUniforms = 0;
    Resources.maxGeometryImageUniforms = 0;
    Resources.maxFragmentImageUniforms = 8;
    Resources.maxCombinedImageUniforms = 8;
    Resources.maxGeometryTextureImageUnits = 16;
    Resources.maxGeometryOutputVertices = 256;
    Resources.maxGeometryTotalOutputComponents = 1024;
    Resources.maxGeometryUniformComponents = 1024;
    Resources.maxGeometryVaryingComponents = 64;
    Resources.maxTessControlInputComponents = 128;
    Resources.maxTessControlOutputComponents = 128;
    Resources.maxTessControlTextureImageUnits = 16;
    Resources.maxTessControlUniformComponents = 1024;
    Resources.maxTessControlTotalOutputComponents = 4096;
    Resources.maxTessEvaluationInputComponents = 128;
    Resources.maxTessEvaluationOutputComponents = 128;
    Resources.maxTessEvaluationTextureImageUnits = 16;
    Resources.maxTessEvaluationUniformComponents = 1024;
    Resources.maxTessPatchComponents = 120;
    Resources.maxPatchVertices = 32;
    Resources.maxTessGenLevel = 64;
    Resources.maxViewports = 16;
    Resources.maxVertexAtomicCounters = 0;
    Resources.maxTessControlAtomicCounters = 0;
    Resources.maxTessEvaluationAtomicCounters = 0;
    Resources.maxGeometryAtomicCounters = 0;
    Resources.maxFragmentAtomicCounters = 8;
    Resources.maxCombinedAtomicCounters = 8;
    Resources.maxAtomicCounterBindings = 1;
    Resources.maxVertexAtomicCounterBuffers = 0;
    Resources.maxTessControlAtomicCounterBuffers = 0;
    Resources.maxTessEvaluationAtomicCounterBuffers = 0;
    Resources.maxGeometryAtomicCounterBuffers = 0;
    Resources.maxFragmentAtomicCounterBuffers = 1;
    Resources.maxCombinedAtomicCounterBuffers = 1;
    Resources.maxAtomicCounterBufferSize = 16384;
    Resources.maxTransformFeedbackBuffers = 4;
    Resources.maxTransformFeedbackInterleavedComponents = 64;
    Resources.maxCullDistances = 8;
    Resources.maxCombinedClipAndCullDistances = 8;
    Resources.maxSamples = 4;
    Resources.maxMeshOutputVerticesNV = 256;
    Resources.maxMeshOutputPrimitivesNV = 512;
    Resources.maxMeshWorkGroupSizeX_NV = 32;
    Resources.maxMeshWorkGroupSizeY_NV = 1;
    Resources.maxMeshWorkGroupSizeZ_NV = 1;
    Resources.maxTaskWorkGroupSizeX_NV = 32;
    Resources.maxTaskWorkGroupSizeY_NV = 1;
    Resources.maxTaskWorkGroupSizeZ_NV = 1;
    Resources.maxMeshViewCountNV = 4;

    Resources.limits.nonInductiveForLoops = true;
    Resources.limits.whileLoops = true;
    Resources.limits.doWhileLoops = true;
    Resources.limits.generalUniformIndexing = true;
    Resources.limits.generalAttributeMatrixVectorIndexing = true;
    Resources.limits.generalVaryingIndexing = true;
    Resources.limits.generalSamplerIndexing = true;
    Resources.limits.generalVariableIndexing = true;
    Resources.limits.generalConstantMatrixVectorIndexing = true;

    // Ten fields this table never set, left at 0 by the value-initialisation
    // above. Nine are mesh-shader limits that glslang only reads when a shader
    // asks for them, so 0 was harmless. maxDualSourceDrawBuffersEXT was not:
    // glslang emits
    //     mediump vec4 gl_SecondaryFragDataEXT[gl_MaxDualSourceDrawBuffersEXT];
    // into the ESSL built-in block, and an array sized 0 fails to parse -- which
    // fails the whole built-in table, so every shader routed through glslang was
    // rejected with "unsupported shader version". It only showed on a context
    // whose ESSL version is below the shader's, since a shader the driver can
    // take is passed straight through; ANGLE presents ES 3.1, so turning ANGLE on
    // meant nothing using #version 320 es could compile at all.
    //
    // The values are glslang's own defaults (glslang/ResourceLimits.cpp).
    Resources.maxDualSourceDrawBuffersEXT = 1;
    Resources.maxMeshOutputVerticesEXT = 256;
    Resources.maxMeshOutputPrimitivesEXT = 256;
    Resources.maxMeshWorkGroupSizeX_EXT = 128;
    Resources.maxMeshWorkGroupSizeY_EXT = 128;
    Resources.maxMeshWorkGroupSizeZ_EXT = 128;
    Resources.maxTaskWorkGroupSizeX_EXT = 128;
    Resources.maxTaskWorkGroupSizeY_EXT = 128;
    Resources.maxTaskWorkGroupSizeZ_EXT = 128;
    Resources.maxMeshViewCountEXT = 4;

    return Resources;
}

int getGLSLVersion(const char* glsl_code) {
    std::string code(glsl_code);
    static std::regex version_pattern(R"(#version\s+(\d{3}))");
    std::smatch match;
    if (std::regex_search(code, match, version_pattern)) {
        return std::stoi(match[1].str());
    }

    return -1;
}

std::string forceSupporterOutput(const std::string& glslCode) {
    bool hasPrecisionFloat =
        glslCode.find("precision ") != std::string::npos && glslCode.find("float;") != std::string::npos;
    bool hasPrecisionInt =
        glslCode.find("precision ") != std::string::npos && glslCode.find("int;") != std::string::npos;

    std::string result = glslCode;
    std::string precisionFloat;
    std::string precisionInt;

    if (hasPrecisionFloat && hasPrecisionInt) {
        std::istringstream iss(result);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(iss, line)) {
            bool isPrecisionLine = (line.find("precision ") != std::string::npos) &&
                                   (line.find("float;") != std::string::npos || line.find("int;") != std::string::npos);
            if (!isPrecisionLine) {
                lines.push_back(line);
            }
        }
        result.clear();
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i != 0) result += '\n';
            result += lines[i];
        }
        precisionFloat = "precision highp float;\n";
        precisionInt = "precision highp int;\n";
    } else {
        precisionFloat = hasPrecisionFloat ? "" : "precision highp float;\n";
        precisionInt = hasPrecisionInt ? "" : "precision highp int;\n";
    }

    size_t lastExtensionPos = result.rfind("#extension");
    size_t insertionPos = 0;

    if (lastExtensionPos != std::string::npos) {
        size_t nextNewline = result.find('\n', lastExtensionPos);
        if (nextNewline != std::string::npos) {
            insertionPos = nextNewline + 1;
        } else {
            insertionPos = result.length();
        }
    } else {
        size_t firstNewline = result.find('\n');
        if (firstNewline != std::string::npos) {
            insertionPos = firstNewline + 1;
        } else {
            result = precisionFloat + precisionInt + result;
            return result;
        }
    }

    result.insert(insertionPos, precisionFloat + precisionInt);
    return result;
}

std::string removeLayoutBinding(const std::string& glslCode) {
    static std::regex bindingRegex(R"(layout\s*\(\s*binding\s*=\s*\d+\s*\)\s*)");
    std::string result = std::regex_replace(glslCode, bindingRegex, "");
    static std::regex bindingRegex2(R"(layout\s*\(\s*binding\s*=\s*\d+\s*,)");
    result = std::regex_replace(result, bindingRegex2, "layout(");
    return result;
}

void trim(std::string& str) {
    str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](int ch) { return !std::isspace(ch); }));
    str.erase(std::find_if(str.rbegin(), str.rend(), [](int ch) { return !std::isspace(ch); }).base(), str.end());
}

// Process all uniform declarations into `uniform <precision> <type> <name>;` form
std::string process_uniform_declarations(const std::string& glslCode) {
    std::string result;
    size_t scan_pos = 0;
    size_t chunk_start = 0;
    const size_t length = glslCode.length();
    const std::vector<std::string> precision_kws = {"highp", "lowp", "mediump"};

    result.reserve(glslCode.length());

    while (scan_pos < length) {
        // Do not parse tokens inside comments. Besides being more correct, this
        // prevents shader comments containing words such as "uniform" or
        // "uniformly" from being mistaken for declarations.
        if (scan_pos + 1 < length && glslCode[scan_pos] == '/' && glslCode[scan_pos + 1] == '/') {
            const size_t newline = glslCode.find('\n', scan_pos + 2);
            scan_pos = (newline == std::string::npos) ? length : newline + 1;
            continue;
        }
        if (scan_pos + 1 < length && glslCode[scan_pos] == '/' && glslCode[scan_pos + 1] == '*') {
            const size_t endComment = glslCode.find("*/", scan_pos + 2);
            scan_pos = (endComment == std::string::npos) ? length : endComment + 2;
            continue;
        }

        // Only treat "uniform" as the GLSL keyword. Snapshot 9 uses generated
        // identifiers such as "_uniform"; matching the substring inside an
        // identifier corrupts the shader and produces "_uniform undeclared".
        const bool uniformKeyword =
            glslCode.compare(scan_pos, 7, "uniform") == 0 &&
            (scan_pos == 0 ||
             !(std::isalnum(static_cast<unsigned char>(glslCode[scan_pos - 1])) ||
               glslCode[scan_pos - 1] == '_')) &&
            (scan_pos + 7 >= length ||
             !(std::isalnum(static_cast<unsigned char>(glslCode[scan_pos + 7])) ||
               glslCode[scan_pos + 7] == '_'));

        if (uniformKeyword) {
            if (scan_pos > chunk_start) {
                result.append(glslCode, chunk_start, scan_pos - chunk_start);
            }

            const size_t decl_start = scan_pos;
            scan_pos += 7; // Skip "uniform"

            std::string precision, type;
            bool found_precision = false;

            while (scan_pos < length) {
                while (scan_pos < length && std::isspace(glslCode[scan_pos]))
                    ++scan_pos;

                for (const auto& kw : precision_kws) {
                    if (glslCode.compare(scan_pos, kw.length(), kw) == 0) {
                        precision = " " + kw;
                        scan_pos += kw.length();
                        found_precision = true;
                        break;
                    }
                }
                if (found_precision) break;

                const size_t type_start = scan_pos;
                while (scan_pos < length && (std::isalnum(glslCode[scan_pos]) || glslCode[scan_pos] == '_')) {
                    ++scan_pos;
                }
                type = glslCode.substr(type_start, scan_pos - type_start);
                break;
            }

            while (scan_pos < length) {
                while (scan_pos < length && std::isspace(glslCode[scan_pos]))
                    ++scan_pos;

                bool found = false;
                for (const auto& kw : precision_kws) {
                    if (glslCode.compare(scan_pos, kw.length(), kw) == 0) {
                        if (precision.empty()) precision = " " + kw;
                        scan_pos += kw.length();
                        found = true;
                        break;
                    }
                }
                if (!found) break;
            }

            if (type.empty()) {
                const size_t type_start = scan_pos;
                while (scan_pos < length && (std::isalnum(glslCode[scan_pos]) || glslCode[scan_pos] == '_')) {
                    ++scan_pos;
                }
                type = glslCode.substr(type_start, scan_pos - type_start);
            }

            while (scan_pos < length && std::isspace(glslCode[scan_pos]))
                ++scan_pos;

            // SPIRV-Cross can emit uniform blocks, for example:
            //
            //   uniform _uniform {
            //       mat4 _instance_00_00;
            //   } _uniform;
            //
            // The old simple-uniform parser treated the first member semicolon
            // as the end of the declaration and destroyed the block. Snapshot 9
            // terrain shaders then referenced an undeclared `_uniform` block.
            // Preserve block declarations byte-for-byte.
            if (scan_pos < length && glslCode[scan_pos] == '{') {
                size_t pos = scan_pos;
                int braceDepth = 0;
                bool inLineComment = false;
                bool inBlockComment = false;

                while (pos < length) {
                    if (inLineComment) {
                        if (glslCode[pos] == '\n') inLineComment = false;
                        ++pos;
                        continue;
                    }
                    if (inBlockComment) {
                        if (pos + 1 < length && glslCode[pos] == '*' && glslCode[pos + 1] == '/') {
                            inBlockComment = false;
                            pos += 2;
                        } else {
                            ++pos;
                        }
                        continue;
                    }
                    if (pos + 1 < length && glslCode[pos] == '/' && glslCode[pos + 1] == '/') {
                        inLineComment = true;
                        pos += 2;
                        continue;
                    }
                    if (pos + 1 < length && glslCode[pos] == '/' && glslCode[pos + 1] == '*') {
                        inBlockComment = true;
                        pos += 2;
                        continue;
                    }

                    if (glslCode[pos] == '{') {
                        ++braceDepth;
                    } else if (glslCode[pos] == '}') {
                        --braceDepth;
                        if (braceDepth == 0) {
                            ++pos; // include closing brace
                            break;
                        }
                    }
                    ++pos;
                }

                // Include optional instance name / array suffix up to the block's
                // terminating semicolon.
                size_t blockEnd = glslCode.find(';', pos);
                if (blockEnd == std::string::npos) {
                    blockEnd = length;
                } else {
                    ++blockEnd;
                }

                result.append(glslCode, decl_start, blockEnd - decl_start);
                scan_pos = chunk_start = blockEnd;
                continue;
            }

            const size_t name_start = scan_pos;
            while (scan_pos < length && (std::isalnum(glslCode[scan_pos]) || glslCode[scan_pos] == '_')) {
                ++scan_pos;
            }
            const std::string name = glslCode.substr(name_start, scan_pos - name_start);

            size_t decl_end = glslCode.find(';', scan_pos);
            if (decl_end == std::string::npos)
                decl_end = length;
            else
                ++decl_end;
            const bool has_initializer = (glslCode.find('=', scan_pos) < decl_end);
            if (has_initializer) {
                result.append("uniform").append(precision).append(" ").append(type).append(" ").append(name).append(
                    ";");
            } else {
                result.append(glslCode, decl_start, decl_end - decl_start);
            }

            scan_pos = chunk_start = decl_end;
        } else {
            ++scan_pos;
        }
    }

    if (chunk_start < length) {
        result.append(glslCode, chunk_start, length - chunk_start);
    }

    return result;
}

std::string processOutColorLocations(const std::string& glslCode) {
    const static std::regex pattern(R"(\n(out highp vec4 outColor)(\d+);)");
    const std::string replacement = "\nlayout(location=$2) $1$2;";
    return std::regex_replace(glslCode, pattern, replacement);
}


// Some Mali GLES compilers reject dynamically indexed fragment-output arrays,
// even when the ESSL source produced by SPIRV-Cross is otherwise valid.  This
// is exposed by Minecraft 26.3's OIT transmittance shaders as:
//
//   S0015: Outputs declared as arrays may only be indexed by a constant
//          integral expression.
//
// Adreno accepts the generated shader, so keep this workaround Mali-only.  We
// deliberately patch the final ESSL rather than Minecraft's input GLSL: this is
// the exact representation the GLES driver sees, and it also covers future
// callers that hit the same SPIRV-Cross lowering pattern.
static bool isMaliRendererForOutputArrayWorkaround() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;

    if (!GLES.glGetString) return false;
    const GLubyte* rendererBytes = GLES.glGetString(GL_RENDERER);
    if (!rendererBytes) return false; // Context may not be current yet; retry later.

    std::string renderer(reinterpret_cast<const char*>(rendererBytes));
    std::string lowered = renderer;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    cached = lowered.find("mali") != std::string::npos ? 1 : 0;
    if (cached == 1) {
        LOG_I("[Shader] Mali renderer detected; dynamic fragment output-array compatibility enabled (%s).",
              renderer.c_str())
    }
    return cached == 1;
}

static std::string trimCopy(const std::string& value) {
    const size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

static bool isMaliIdentStart(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return std::isalpha(u) || c == '_';
}

static bool isMaliIdentChar(char c) {
    const unsigned char u = static_cast<unsigned char>(c);
    return std::isalnum(u) || c == '_';
}

static bool isIntegralConstantExpression(const std::string& expression) {
    std::string value = trimCopy(expression);
    if (value.empty()) return false;

    size_t i = 0;
    if (value[i] == '+' || value[i] == '-') {
        if (++i >= value.size()) return false;
    }

    bool hex = false;
    if (i + 2 <= value.size() && value[i] == '0' && i + 1 < value.size() &&
        (value[i + 1] == 'x' || value[i + 1] == 'X')) {
        hex = true;
        i += 2;
        if (i >= value.size()) return false;
    }

    size_t digits = 0;
    for (; i < value.size(); ++i) {
        const char c = value[i];
        if (c == 'u' || c == 'U') {
            return digits > 0 && i + 1 == value.size();
        }
        const unsigned char u = static_cast<unsigned char>(c);
        const bool valid = hex ? std::isxdigit(u) != 0 : std::isdigit(u) != 0;
        if (!valid) return false;
        ++digits;
    }
    return digits > 0;
}

static bool parseMaliPositiveExtentLiteral(const std::string& expression, unsigned int* valueOut) {
    std::string value = trimCopy(expression);
    if (value.empty()) return false;
    if (value.back() == 'u' || value.back() == 'U') value.pop_back();
    if (value.empty()) return false;
    for (char c : value) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    try {
        const unsigned long parsed = std::stoul(value);
        if (parsed == 0 || parsed > 32) return false;
        if (valueOut) *valueOut = static_cast<unsigned int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static bool isIdentifierBoundary(const std::string& text, size_t pos) {
    return pos >= text.size() || !isMaliIdentChar(text[pos]);
}

static unsigned int resolveMaliNamedExtent(const std::string& essl, const std::string& name) {
    if (name.empty() || !isMaliIdentStart(name[0])) return 0;
    for (char c : name) if (!isMaliIdentChar(c)) return 0;

    size_t pos = 0;
    while ((pos = essl.find(name, pos)) != std::string::npos) {
        const bool leftOk = pos == 0 || !isMaliIdentChar(essl[pos - 1]);
        const bool rightOk = isIdentifierBoundary(essl, pos + name.size());
        if (!leftOk || !rightOk) {
            pos += name.size();
            continue;
        }

        const size_t lineStartPos = essl.rfind('\n', pos);
        const size_t lineStart = lineStartPos == std::string::npos ? 0 : lineStartPos + 1;
        const size_t lineEndPos = essl.find('\n', pos);
        const size_t lineEnd = lineEndPos == std::string::npos ? essl.size() : lineEndPos;
        const std::string line = essl.substr(lineStart, lineEnd - lineStart);
        const size_t localName = pos - lineStart;

        // #define NAME 4
        const size_t definePos = line.find("#define");
        if (definePos != std::string::npos && definePos < localName) {
            size_t rhs = localName + name.size();
            while (rhs < line.size() && std::isspace(static_cast<unsigned char>(line[rhs]))) ++rhs;
            size_t rhsEnd = rhs;
            while (rhsEnd < line.size() && !std::isspace(static_cast<unsigned char>(line[rhsEnd]))) ++rhsEnd;
            unsigned int parsed = 0;
            if (parseMaliPositiveExtentLiteral(line.substr(rhs, rhsEnd - rhs), &parsed)) return parsed;
        }

        // const int/uint NAME = 4;
        const size_t constPos = line.find("const");
        const size_t eqPos = line.find('=', localName + name.size());
        if (constPos != std::string::npos && constPos < localName && eqPos != std::string::npos) {
            size_t rhs = eqPos + 1;
            while (rhs < line.size() && std::isspace(static_cast<unsigned char>(line[rhs]))) ++rhs;
            size_t rhsEnd = rhs;
            while (rhsEnd < line.size() && line[rhsEnd] != ';' &&
                   !std::isspace(static_cast<unsigned char>(line[rhsEnd]))) ++rhsEnd;
            unsigned int parsed = 0;
            if (parseMaliPositiveExtentLiteral(line.substr(rhs, rhsEnd - rhs), &parsed)) return parsed;
        }

        pos += name.size();
    }
    return 0;
}

static bool containsMaliTokenInRange(const std::string& text, size_t begin, size_t end,
                                     const std::string& token);

static unsigned int resolveMaliOutputArrayExtent(const std::string& essl, const std::string& rawExpression) {
    const std::string expression = trimCopy(rawExpression);
    if (expression.empty()) return 0;

    unsigned int literal = 0;
    if (parseMaliPositiveExtentLiteral(expression, &literal)) return literal;

    if (expression == "gl_MaxDrawBuffers") {
        GLint maxDrawBuffers = 0;
        if (GLES.glGetIntegerv) GLES.glGetIntegerv(GL_MAX_DRAW_BUFFERS, &maxDrawBuffers);
        if (maxDrawBuffers <= 0) maxDrawBuffers = 8;
        if (maxDrawBuffers > 32) maxDrawBuffers = 32;
        return static_cast<unsigned int>(maxDrawBuffers);
    }

    return resolveMaliNamedExtent(essl, expression);
}

// Mali's fragment-output rule is about *runtime* indexing.  A named #define or
// const int is still a compile-time integral constant, even though the simple
// literal parser above cannot recognize it by itself.  Treat those as constants
// so the dynamic-access pass never rewrites array declarations such as
// `out vec4 color[SPIRV_Cross_Combined_MAX];` into an expression.
static bool isMaliCompileTimeConstantIndex(const std::string& essl, const std::string& rawExpression) {
    std::string expression = trimCopy(rawExpression);
    if (expression.empty()) return false;

    // Strip harmless balanced outer parentheses.
    bool changed = true;
    while (changed && expression.size() >= 2 && expression.front() == '(' && expression.back() == ')') {
        changed = false;
        int depth = 0;
        bool wrapsWholeExpression = true;
        for (size_t i = 0; i < expression.size(); ++i) {
            if (expression[i] == '(') ++depth;
            else if (expression[i] == ')') --depth;
            if (depth == 0 && i + 1 < expression.size()) {
                wrapsWholeExpression = false;
                break;
            }
        }
        if (wrapsWholeExpression) {
            expression = trimCopy(expression.substr(1, expression.size() - 2));
            changed = true;
        }
    }

    if (isIntegralConstantExpression(expression)) return true;
    if (expression == "gl_MaxDrawBuffers") return true;
    return resolveMaliNamedExtent(essl, expression) > 0;
}

static bool isMaliOutputArrayDeclarator(const std::string& source, size_t pathStart,
                                        size_t closeBracket, const std::string& indexExpression) {
    // A declaration is the most important false-positive to reject.  SPIRV-Cross
    // commonly emits symbolic extents, and the old scanner interpreted the
    // declaration bracket as one dynamic read.  Do not require us to resolve the
    // extent here; declaration syntax itself is sufficient to prove this is not
    // a runtime array access.
    (void) indexExpression;

    size_t after = closeBracket + 1;
    while (after < source.size() && std::isspace(static_cast<unsigned char>(source[after]))) ++after;
    if (after >= source.size() || (source[after] != ';' && source[after] != ',' && source[after] != '='))
        return false;

    const size_t lineStartPos = source.rfind('\n', pathStart);
    const size_t lineStart = lineStartPos == std::string::npos ? 0 : lineStartPos + 1;
    if (containsMaliTokenInRange(source, lineStart, pathStart, "out")) return true;

    // Interface-block members do not repeat the `out` qualifier on every line.
    const size_t openBrace = source.rfind('{', pathStart);
    const size_t closeBrace = source.rfind('}', pathStart);
    if (openBrace != std::string::npos && (closeBrace == std::string::npos || openBrace > closeBrace)) {
        size_t headerStart = source.rfind(';', openBrace);
        const size_t headerLine = source.rfind('\n', openBrace);
        if (headerStart == std::string::npos || (headerLine != std::string::npos && headerLine > headerStart))
            headerStart = headerLine;
        headerStart = headerStart == std::string::npos ? 0 : headerStart + 1;
        if (containsMaliTokenInRange(source, headerStart, openBrace, "out")) return true;
    }
    return false;
}

static std::string normalizeMaliAccessPath(const std::string& path) {
    std::string out;
    out.reserve(path.size());
    bool pendingSpace = false;
    for (char c : path) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            pendingSpace = true;
            continue;
        }
        if (c == '.') {
            while (!out.empty() && out.back() == ' ') out.pop_back();
            out.push_back('.');
            pendingSpace = false;
            continue;
        }
        if (pendingSpace && !out.empty() && out.back() != '.') out.push_back(' ');
        pendingSpace = false;
        out.push_back(c);
    }
    return trimCopy(out);
}

static bool extractMaliAccessPathBeforeBracket(const std::string& source, size_t bracketPos,
                                                size_t* pathStartOut, std::string* pathOut) {
    if (bracketPos == 0) return false;
    size_t p = bracketPos;
    while (p > 0 && std::isspace(static_cast<unsigned char>(source[p - 1]))) --p;
    const size_t pathEnd = p;
    if (p == 0 || !isMaliIdentChar(source[p - 1])) return false;

    // Parse IDENT(.IDENT)* backwards, allowing whitespace around dots.
    while (true) {
        while (p > 0 && isMaliIdentChar(source[p - 1])) --p;
        if (p == 0) break;
        size_t q = p;
        while (q > 0 && std::isspace(static_cast<unsigned char>(source[q - 1]))) --q;
        if (q == 0 || source[q - 1] != '.') break;
        q--;
        while (q > 0 && std::isspace(static_cast<unsigned char>(source[q - 1]))) --q;
        if (q == 0 || !isMaliIdentChar(source[q - 1])) break;
        p = q;
    }

    std::string path = normalizeMaliAccessPath(source.substr(p, pathEnd - p));
    if (path.empty()) return false;
    // Reject anything that is not IDENT(.IDENT)*.
    bool wantStart = true;
    for (size_t i = 0; i < path.size();) {
        if (wantStart) {
            if (!isMaliIdentStart(path[i])) return false;
            ++i;
            while (i < path.size() && isMaliIdentChar(path[i])) ++i;
            wantStart = false;
        }
        if (i == path.size()) break;
        if (path[i] != '.') return false;
        ++i;
        wantStart = true;
    }
    if (wantStart) return false;

    if (pathStartOut) *pathStartOut = p;
    if (pathOut) *pathOut = path;
    return true;
}

static size_t findMatchingMaliBracket(const std::string& source, size_t openPos) {
    if (openPos >= source.size() || source[openPos] != '[') return std::string::npos;
    int depth = 1;
    for (size_t i = openPos + 1; i < source.size(); ++i) {
        if (source[i] == '[') ++depth;
        else if (source[i] == ']' && --depth == 0) return i;
    }
    return std::string::npos;
}

struct MaliDynamicArrayWrite {
    size_t start = 0;
    size_t end = 0;
    std::string path;
    std::string index;
    std::string suffix;
    std::string op;
    std::string rhs;
};

static std::vector<MaliDynamicArrayWrite> findMaliDynamicArrayWrites(const std::string& source) {
    std::vector<MaliDynamicArrayWrite> writes;
    size_t search = 0;
    while (true) {
        const size_t bracket = source.find('[', search);
        if (bracket == std::string::npos) break;
        search = bracket + 1;

        size_t pathStart = 0;
        std::string path;
        if (!extractMaliAccessPathBeforeBracket(source, bracket, &pathStart, &path)) continue;

        const size_t close = findMatchingMaliBracket(source, bracket);
        if (close == std::string::npos) continue;
        const std::string index = trimCopy(source.substr(bracket + 1, close - bracket - 1));
        if (index.empty() || isMaliCompileTimeConstantIndex(source, index)) continue;

        size_t p = close + 1;
        std::string suffix;
        while (true) {
            while (p < source.size() && std::isspace(static_cast<unsigned char>(source[p]))) ++p;
            if (p >= source.size() || source[p] != '.') break;
            const size_t suffixStart = p++;
            while (p < source.size() && std::isspace(static_cast<unsigned char>(source[p]))) ++p;
            if (p >= source.size() || !isMaliIdentStart(source[p])) { p = suffixStart; break; }
            ++p;
            while (p < source.size() && isMaliIdentChar(source[p])) ++p;
            suffix += normalizeMaliAccessPath(source.substr(suffixStart, p - suffixStart));
        }

        while (p < source.size() && std::isspace(static_cast<unsigned char>(source[p]))) ++p;
        std::string op;
        if (p + 1 < source.size() &&
            ((source[p] == '+' || source[p] == '-' || source[p] == '*' || source[p] == '/') && source[p + 1] == '=')) {
            op = source.substr(p, 2);
            p += 2;
        } else if (p < source.size() && source[p] == '=' && (p + 1 >= source.size() || source[p + 1] != '=')) {
            op = "=";
            ++p;
        } else {
            continue;
        }

        while (p < source.size() && std::isspace(static_cast<unsigned char>(source[p]))) ++p;
        const size_t rhsStart = p;
        int paren = 0, square = 0, brace = 0;
        size_t semicolon = std::string::npos;
        for (; p < source.size(); ++p) {
            const char c = source[p];
            if (c == '(') ++paren;
            else if (c == ')' && paren > 0) --paren;
            else if (c == '[') ++square;
            else if (c == ']' && square > 0) --square;
            else if (c == '{') ++brace;
            else if (c == '}' && brace > 0) --brace;
            else if (c == ';' && paren == 0 && square == 0 && brace == 0) {
                semicolon = p;
                break;
            }
        }
        if (semicolon == std::string::npos) continue;

        writes.push_back({pathStart, semicolon + 1, path, index, suffix, op,
                          trimCopy(source.substr(rhsStart, semicolon - rhsStart))});
        search = semicolon + 1;
    }
    return writes;
}

static std::string finalMaliIdentifier(const std::string& path) {
    const size_t dot = path.find_last_of('.');
    return dot == std::string::npos ? path : path.substr(dot + 1);
}

static unsigned int findMaliArrayExtentForIdentifier(const std::string& essl, const std::string& identifier) {
    if (identifier.empty()) return 0;
    unsigned int extent = 0;
    size_t pos = 0;
    while ((pos = essl.find(identifier, pos)) != std::string::npos) {
        const bool leftOk = pos == 0 || !isMaliIdentChar(essl[pos - 1]);
        const bool rightOk = isIdentifierBoundary(essl, pos + identifier.size());
        if (!leftOk || !rightOk) {
            pos += identifier.size();
            continue;
        }
        size_t p = pos + identifier.size();
        while (p < essl.size() && std::isspace(static_cast<unsigned char>(essl[p]))) ++p;
        if (p >= essl.size() || essl[p] != '[') {
            pos += identifier.size();
            continue;
        }
        const size_t close = findMatchingMaliBracket(essl, p);
        if (close == std::string::npos) break;
        const unsigned int resolved = resolveMaliOutputArrayExtent(essl, essl.substr(p + 1, close - p - 1));
        if (resolved > extent) extent = resolved;
        pos = close + 1;
    }
    return extent;
}


static bool containsMaliTokenInRange(const std::string& text, size_t begin, size_t end,
                                     const std::string& token) {
    if (begin >= end || token.empty()) return false;
    size_t pos = begin;
    while ((pos = text.find(token, pos)) != std::string::npos && pos < end) {
        const bool leftOk = pos == begin || !isMaliIdentChar(text[pos - 1]);
        const size_t after = pos + token.size();
        const bool rightOk = after >= end || !isMaliIdentChar(text[after]);
        if (leftOk && rightOk) return true;
        pos = after;
    }
    return false;
}

static bool isMaliLikelyFragmentOutputArrayIdentifier(const std::string& essl,
                                                       const std::string& identifier) {
    if (identifier == "gl_FragData" || identifier == "gl_SecondaryFragDataEXT" ||
        identifier == "gl_SecondaryFragData") {
        return true;
    }
    if (identifier.empty()) return false;

    size_t pos = 0;
    while ((pos = essl.find(identifier, pos)) != std::string::npos) {
        const bool leftOk = pos == 0 || !isMaliIdentChar(essl[pos - 1]);
        const bool rightOk = isIdentifierBoundary(essl, pos + identifier.size());
        if (!leftOk || !rightOk) {
            pos += identifier.size();
            continue;
        }

        size_t p = pos + identifier.size();
        while (p < essl.size() && std::isspace(static_cast<unsigned char>(essl[p]))) ++p;
        if (p >= essl.size() || essl[p] != '[') {
            pos += identifier.size();
            continue;
        }
        const size_t close = findMatchingMaliBracket(essl, p);
        if (close == std::string::npos ||
            resolveMaliOutputArrayExtent(essl, essl.substr(p + 1, close - p - 1)) == 0) {
            pos += identifier.size();
            continue;
        }

        // Direct fragment output declaration, e.g.
        //   layout(location = 0) out highp vec4 FragColor[8];
        const size_t lineStartPos = essl.rfind('\n', pos);
        const size_t lineStart = lineStartPos == std::string::npos ? 0 : lineStartPos + 1;
        if (containsMaliTokenInRange(essl, lineStart, pos, "out")) return true;

        // SPIRV-Cross can also emit an output interface block.  A member line
        // then has no `out` token itself, so inspect the still-open block header:
        //   out SomeBlock { vec4 color[8]; } outputs;
        const size_t openBrace = essl.rfind('{', pos);
        const size_t closeBrace = essl.rfind('}', pos);
        if (openBrace != std::string::npos &&
            (closeBrace == std::string::npos || openBrace > closeBrace)) {
            size_t headerStart = essl.rfind(';', openBrace);
            const size_t headerLine = essl.rfind('\n', openBrace);
            if (headerStart == std::string::npos ||
                (headerLine != std::string::npos && headerLine > headerStart)) {
                headerStart = headerLine;
            }
            headerStart = headerStart == std::string::npos ? 0 : headerStart + 1;
            if (containsMaliTokenInRange(essl, headerStart, openBrace, "out")) return true;
        }

        pos = close + 1;
    }
    return false;
}

struct MaliDynamicArrayAccess {
    size_t start = 0;
    size_t end = 0;
    std::string path;
    std::string index;
    unsigned int extent = 0;
};

static std::vector<MaliDynamicArrayAccess> findMaliDynamicOutputArrayReads(const std::string& source,
                                                                           size_t* remainingLvaluesOut = nullptr) {
    if (remainingLvaluesOut) *remainingLvaluesOut = 0;
    std::vector<MaliDynamicArrayAccess> reads;
    size_t search = 0;
    while (true) {
        const size_t bracket = source.find('[', search);
        if (bracket == std::string::npos) break;
        search = bracket + 1;

        size_t pathStart = 0;
        std::string path;
        if (!extractMaliAccessPathBeforeBracket(source, bracket, &pathStart, &path)) continue;
        const size_t close = findMatchingMaliBracket(source, bracket);
        if (close == std::string::npos) continue;

        const std::string index = trimCopy(source.substr(bracket + 1, close - bracket - 1));
        if (index.empty()) continue;
        if (isMaliOutputArrayDeclarator(source, pathStart, close, index)) continue;
        if (isMaliCompileTimeConstantIndex(source, index)) continue;

        const std::string identifier = finalMaliIdentifier(path);
        if (!isMaliLikelyFragmentOutputArrayIdentifier(source, identifier)) continue;
        const unsigned int extent = findMaliArrayExtentForIdentifier(source, identifier);
        if (extent == 0 || extent > 32) continue;

        // If this access is still the left side of a direct assignment, leave it
        // for the store pass.  V4 already handles these; counting it here makes
        // any unhandled write obvious in the log rather than turning a ternary
        // into an invalid lvalue.
        size_t p = close + 1;
        while (true) {
            while (p < source.size() && std::isspace(static_cast<unsigned char>(source[p]))) ++p;
            if (p >= source.size() || source[p] != '.') break;
            ++p;
            while (p < source.size() && std::isspace(static_cast<unsigned char>(source[p]))) ++p;
            if (p >= source.size() || !isMaliIdentStart(source[p])) break;
            ++p;
            while (p < source.size() && isMaliIdentChar(source[p])) ++p;
        }
        while (p < source.size() && std::isspace(static_cast<unsigned char>(source[p]))) ++p;
        const bool directAssignment =
            (p < source.size() && source[p] == '=' && (p + 1 >= source.size() || source[p + 1] != '=')) ||
            (p + 1 < source.size() &&
             (source[p] == '+' || source[p] == '-' || source[p] == '*' || source[p] == '/') &&
             source[p + 1] == '=');
        if (directAssignment) {
            if (remainingLvaluesOut) ++(*remainingLvaluesOut);
            continue;
        }

        reads.push_back({pathStart, close + 1, path, index, extent});
        search = close + 1;
    }
    return reads;
}

static std::string applyMaliFragmentOutputArrayReadWorkaround(const std::string& essl,
                                                               size_t* rewritesOut = nullptr,
                                                               size_t* remainingLvaluesOut = nullptr) {
    if (rewritesOut) *rewritesOut = 0;
    if (remainingLvaluesOut) *remainingLvaluesOut = 0;

    const auto reads = findMaliDynamicOutputArrayReads(essl, remainingLvaluesOut);
    if (reads.empty()) return essl;

    // Replace from the end so source positions remain valid.  Every branch uses
    // a literal index, satisfying Mali's fragment-output restriction without
    // changing the output declaration/location semantics.
    std::string patched = essl;
    size_t count = 0;
    for (auto it = reads.rbegin(); it != reads.rend(); ++it) {
        const auto& access = *it;
        std::string replacement = "(";
        const std::string idxExpr = "int(" + access.index + ")";
        for (unsigned int i = 0; i < access.extent; ++i) {
            replacement += "(" + idxExpr + " == " + std::to_string(i) + ") ? " +
                           access.path + "[" + std::to_string(i) + "] : ";
        }
        // Out-of-range fragment-output indexing is undefined anyway.  Use the
        // first element only as a syntactically valid terminal expression.
        replacement += access.path + "[0])";
        patched.replace(access.start, access.end - access.start, replacement);
        ++count;
    }

    if (rewritesOut) *rewritesOut = count;
    if (count > 0) {
        LOG_W_FORCE("[Shader] Mali fragment output-array workaround rewrote %zu dynamic read(s).", count)
    }
    return patched;
}

static std::string applyMaliFragmentOutputArrayWorkaround(const std::string& essl,
                                                           GLenum shaderType,
                                                           size_t* rewritesOut = nullptr) {
    if (rewritesOut) *rewritesOut = 0;
    if (shaderType != GL_FRAGMENT_SHADER || !isMaliRendererForOutputArrayWorkaround()) return essl;

    try {
        const auto writes = findMaliDynamicArrayWrites(essl);
        if (writes.empty()) return essl;

        std::string patched;
        patched.reserve(essl.size() + writes.size() * 256);
        size_t cursor = 0;
        unsigned int serial = 0;
        size_t rewriteCount = 0;
        size_t candidateCount = 0;

        for (const auto& write : writes) {
            if (write.start < cursor) continue;
            const unsigned int extent = findMaliArrayExtentForIdentifier(essl, finalMaliIdentifier(write.path));
            if (extent == 0 || extent > 32) continue;
            ++candidateCount;

            patched.append(essl, cursor, write.start - cursor);
            const std::string temp = "_mg_mali_out_idx_" + std::to_string(serial++);
            patched += "{ int " + temp + " = int(" + write.index + "); ";
            for (unsigned int i = 0; i < extent; ++i) {
                patched += (i == 0 ? "if (" : " else if (");
                patched += temp + " == " + std::to_string(i) + ") " + write.path + "[" +
                           std::to_string(i) + "]" + write.suffix + " " + write.op + " " + write.rhs + ";";
            }
            patched += " }";
            cursor = write.end;
            ++rewriteCount;
        }

        if (rewriteCount == 0) return essl;
        patched.append(essl, cursor, std::string::npos);
        if (rewritesOut) *rewritesOut = rewriteCount;
        LOG_W_FORCE("[Shader] Mali dynamic-array fallback found %zu candidate array(s).", candidateCount)
        LOG_W_FORCE("[Shader] Mali fragment output-array workaround rewrote %zu dynamic store(s).", rewriteCount)
        return patched;
    } catch (const std::exception& error) {
        // Never allow a compatibility rewrite to take the renderer process down.
        LOG_E("[Shader] Mali output-array workaround skipped after parser exception: %s", error.what())
        return essl;
    } catch (...) {
        LOG_E("[Shader] Mali output-array workaround skipped after unknown parser exception")
        return essl;
    }
}

// Compatibility fallback for desktop GLSL 1.50.
//
// The normal MobileGlues path is still GLSL -> glslang/SPIR-V -> SPIRV-Cross
// -> ESSL.  Some Minecraft post-processing shaders (notably The Broken Script
// 2.0.3) are valid desktop GLSL 1.50 but can be rejected by that path.  The old
// failure behaviour returned the original "#version 150" source to ANGLE,
// which is an ES compiler.  ANGLE then reports:
//   '150' : client/version number not supported
// and the post chain can render the world black.
//
// Only use this source-level ESSL 3.00 fallback after the normal converter has
// failed, and only for GLSL 1.50.  This keeps existing shader-pack behaviour
// untouched.
static std::string glsl150CompatFallback(const char* glslCode, GLenum glslType, uint esslVersion) {
    if (glslCode == nullptr) return "";

    std::string result(glslCode);

    static const std::regex version150(
        R"(#version\s+150(?:\s+(?:core|compatibility))?)",
        std::regex::ECMAScript);

    if (!std::regex_search(result, version150)) {
        return "";
    }

    // Both stages attached to a GLES program must use the same ESSL version.
    // Do not hard-code 300 here: the normal SPIRV-Cross path targets the
    // device's active ESSL level (for example 320 on ES 3.2). A 300 fallback
    // paired with a successfully converted 320 shader compiles separately but
    // fails at link time with "shader version mismatch".
    const std::string targetVersion =
        "#version " + std::to_string(esslVersion) + " es";
    result = std::regex_replace(
        result,
        version150,
        targetVersion,
        std::regex_constants::format_first_only);

    // Match the cleanup already performed by the normal SPIRV-Cross path.
    result = removeLayoutBinding(result);

    // Common compatibility-profile texture intrinsics have generic ESSL 3.00
    // equivalents.  Do not touch already-modern texture() calls.
    result = std::regex_replace(result, std::regex(R"(\btexture2D\s*\()"), "texture(");
    result = std::regex_replace(result, std::regex(R"(\btextureCube\s*\()"), "texture(");
    result = std::regex_replace(result, std::regex(R"(\btexture2DLod\s*\()"), "textureLod(");
    result = std::regex_replace(result, std::regex(R"(\btextureCubeLod\s*\()"), "textureLod(");

    // ESSL fragment/vertex shaders require explicit precision.  Reuse the
    // existing MobileGlues helper so we do not duplicate precision lines.
    // Desktop drivers are more permissive than ESSL 3.00 about a few
    // expressions used by Minecraft mods. Normalize the common forms seen in
    // The Broken Script before handing the fallback source to ANGLE.

    // bvec -> vec conversion used by the mod's 4D simplex noise helper.
    result = std::regex_replace(
        result,
        std::regex(R"(vec4\s*\(\s*lessThan\s*\(\s*p\s*,\s*vec4\s*\(\s*0\.0\s*\)\s*\)\s*\))"),
        "mix(vec4(0.0), vec4(1.0), lessThan(p, vec4(0.0)))");

    // textureSize() returns ivecN. ESSL requires an explicit conversion before
    // multiplying it by a floating scalar.
    result = std::regex_replace(
        result,
        std::regex(R"(textureSize\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*0\s*\)\s*\*\s*([0-9]*\.[0-9]+))"),
        "vec2(textureSize($1, 0)) * $2");

    // Desktop GLSL accepts arithmetic such as:
    //   vec2Value - 1
    // while strict ESSL rejects the int scalar against a float vector.
    // Broken Script's CalcEyeFromWindow() uses exactly this form.  Normalize
    // integer scalar literals adjacent to vector-style expressions to floats.
    result = std::regex_replace(
        result,
        std::regex(R"((\)\s*-\s*)1(\s*;))"),
        "$1 1.0$2");
    result = std::regex_replace(
        result,
        std::regex(R"((\]\s*-\s*)1(\s*;))"),
        "$1 1.0$2");

    result = forceSupporterOutput(result);

    // Preserve existing MobileGlues output-location normalization.
    result = processOutColorLocations(result);

    // GLSL 1.50 core uses explicit fragment outputs, but compatibility shaders
    // may still contain gl_FragColor.  ESSL 3.00 removed that built-in.
    if (glslType == GL_FRAGMENT_SHADER &&
        result.find("gl_FragColor") != std::string::npos) {
        result = std::regex_replace(
            result, std::regex(R"(\bgl_FragColor\b)"), "mg_FragColor");

        const size_t firstNewline = result.find('\n');
        const size_t insertPos =
            (firstNewline == std::string::npos) ? 0 : firstNewline + 1;
        result.insert(insertPos,
                      "layout(location = 0) out highp vec4 mg_FragColor;\n");
    }

    return result;
}

std::string GLSLtoGLSLES(const char* glsl_code, GLenum glsl_type, uint essl_version, uint glsl_version,
                         int& return_code, std::string* mali_store_only_fallback) {
    if (mali_store_only_fallback) mali_store_only_fallback->clear();
    const bool maliOutputArrayWorkaround =
        glsl_type == GL_FRAGMENT_SHADER && isMaliRendererForOutputArrayWorkaround();

    std::string sha256_string(glsl_code);
    sha256_string += "\n//" + std::to_string(MAJOR) + "." + std::to_string(MINOR) + "." + std::to_string(REVISION) +
                     "|" + std::to_string(essl_version) + "|compat-s9ubo-bs150-v7-version-match";
    if (maliOutputArrayWorkaround) sha256_string += "|mali-frag-out-array-v7-decl-safe";
    const std::string maliFallbackCacheKey = sha256_string + "|store-only-fallback";
    const char* cachedESSL = Cache::get_instance().get(sha256_string.c_str());
    if (cachedESSL) {
        if (mali_store_only_fallback && maliOutputArrayWorkaround) {
            const char* cachedFallback = Cache::get_instance().get(maliFallbackCacheKey.c_str());
            if (cachedFallback) *mali_store_only_fallback = cachedFallback;
        }
        LOG_D("GLSL Hit Cache:\n%s\n-->\n%s", glsl_code, cachedESSL)
        return_code = 0;
        return (char*)cachedESSL;
    }

    return_code = -1;
    // std::string converted = glsl_version<140? GLSLtoGLSLES_1(glsl_code, glsl_type, essl_version,
    // return_code):GLSLtoGLSLES_2(glsl_code, glsl_type, essl_version, return_code);
    std::string converted = GLSLtoGLSLES_2(glsl_code, glsl_type, essl_version, return_code);

    // Never pass a failed desktop GLSL 1.50 conversion straight through to an
    // ES/ANGLE compiler.  Try the conservative ESSL 3.00 compatibility path.
    if ((return_code < 0 || converted.empty()) && glsl_version == 150) {
        LOG_W_FORCE("[Shader] GLSL 150 SPIR-V conversion failed; trying ESSL 300 compatibility fallback.")
        converted = glsl150CompatFallback(glsl_code, glsl_type, essl_version);
        if (!converted.empty()) {
            return_code = 1;
            LOG_W_FORCE("[Shader] GLSL 150 compatibility fallback generated ESSL %u source.", essl_version)
        }
    }

    if (return_code >= 0 && !converted.empty()) {
        converted = process_uniform_declarations(converted);

        // Snapshot 5+ compatibility. `_uniform` may be generated by
        // SPIRV-Cross after the launcher has already had its chance to rename
        // source identifiers. Rename the complete GLSL identifier here, in the
        // translated ESSL, so declarations and all references remain consistent.
        converted = std::regex_replace(
            converted,
            std::regex(R"(\b_uniform\b)"),
            "_db_uvar");

        std::string maliStoreOnlyFallback;
        if (maliOutputArrayWorkaround) {
            size_t storeRewrites = 0;
            converted = applyMaliFragmentOutputArrayWorkaround(converted, glsl_type, &storeRewrites);

            // Keep the store-only form before the newer dynamic-read rewrite.
            // Some Mali compilers reject a ternary-expanded output-array read when
            // the surrounding GLSL context still requires an lvalue (for example
            // an out/inout argument). shader.cpp can retry this known-good form
            // instead of lying about GL_COMPILE_STATUS and linking a broken program.
            maliStoreOnlyFallback = converted;

            size_t readRewrites = 0;
            size_t remainingLvalues = 0;
            converted = applyMaliFragmentOutputArrayReadWorkaround(
                converted, &readRewrites, &remainingLvalues);
            if (remainingLvalues > 0) {
                LOG_E("[Shader] Mali fragment output-array workaround still has %zu dynamic lvalue access(es).",
                      remainingLvalues)
            }
            if (readRewrites == 0) maliStoreOnlyFallback.clear();
            if (mali_store_only_fallback && !maliStoreOnlyFallback.empty()) {
                *mali_store_only_fallback = maliStoreOnlyFallback;
            }
        }

        Cache::get_instance().put(sha256_string.c_str(), converted.c_str());
        if (!maliStoreOnlyFallback.empty()) {
            Cache::get_instance().put(maliFallbackCacheKey.c_str(), maliStoreOnlyFallback.c_str());
        }
        return converted;
    }

    // Preserve historical behaviour for other GLSL versions.  The known-bad
    // GLSL 1.50 -> ANGLE pass-through is handled above.
    return glsl_code;
}

std::string replace_line_starting_with(const std::string& glslCode, const std::string& starting,
                                       const std::string& substitution = "") {
    std::string result;
    size_t length = glslCode.size();
    size_t start = 0;
    size_t current = 0;

    auto append_chunk = [&](size_t end) {
        if (end > start) {
            result.append(glslCode, start, end - start);
        }
    };

    while (current < length) {
        // Skip whitespace at line begin
        size_t lineStart = current;
        while (current < length && (glslCode[current] == ' ' || glslCode[current] == '\t')) {
            current++;
        }

        // Check whether #line directive
        bool isLineDirective = false;
        if (current + 5 <= length && glslCode.compare(current, 5, "#line") == 0) {
            isLineDirective = true;
        }

        // Move to line end
        while (current < length && glslCode[current] != '\r' && glslCode[current] != '\n') {
            current++;
        }

        // Handle carriage return
        size_t newlineLength = 0;
        if (current < length) {
            if (glslCode[current] == '\r') {
                newlineLength = (current + 1 < length && glslCode[current + 1] == '\n') ? 2 : 1;
            } else {
                newlineLength = 1;
            }
        }

        if (isLineDirective) {
            // Find #line directive ->
            //  1. Append chunk
            append_chunk(lineStart); // from chunk_begin to before `#line`
            // 2. Skip this line (incl. \n)
            current += newlineLength;
            start = current; // 3. Starting from next line

            result += substitution;
        } else {
            // move to a new line
            current += newlineLength;
        }
    }

    // append last block
    append_chunk(current);
    return result;
}

static inline void replace_all(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

static size_t find_insertion_point(const std::string& glsl) {
    size_t pos = 0;
    size_t insertion_point = 0;

    size_t version_pos = glsl.find("#version");
    if (version_pos != std::string::npos) {
        size_t version_end = glsl.find('\n', version_pos);
        if (version_end == std::string::npos) {
            version_end = glsl.length();
        } else {
            version_end++;
        }
        insertion_point = version_end;
        pos = version_end;
    } else {
        insertion_point = 0;
        pos = 0;
    }

    while (pos < glsl.length()) {
        size_t line_begin = pos;
        while (pos < glsl.length() && std::isspace(glsl[pos])) {
            pos++;
        }
        if (pos >= glsl.length()) break;

        if (glsl[pos] == '#') {
            pos++;
            while (pos < glsl.length() && std::isspace(glsl[pos])) {
                pos++;
            }
            if (glsl.compare(pos, 9, "extension") == 0) {
                size_t ext_end = glsl.find('\n', pos);
                if (ext_end == std::string::npos) {
                    ext_end = glsl.length();
                } else {
                    ext_end++;
                }
                insertion_point = ext_end;
                pos = ext_end;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    return insertion_point;
}

void process_sampler_buffer(std::string& source) { // a simplized version, should be rewritten in the future
    if (source.find("isamplerBuffer") == std::string::npos) {
        return;
    }

    size_t pos = 0;
    while ((pos = source.find("isamplerBuffer", pos)) != std::string::npos) {
        source.replace(pos, 14, "isampler2D");
        pos += 11;
    }

    std::regex pattern(R"(texelFetch\s*\(\s*(\w+)\s*,\s*([^)]+?)\s*\))");
    source = std::regex_replace(source, pattern,
                                "texelFetch($1, ivec2(($2) % u_BufferTexWidth, ($2) / u_BufferTexWidth), 0)");

    const char* boundaryProtection = R"(
ivec2 bufferCoords(int index) {
    int width = u_BufferTexWidth;
    int x = index % width;
    int y = index / width;
    if (y >= u_BufferTexHeight) {
        y = u_BufferTexHeight - 1;
        x = width - 1;
    }
    return ivec2(x, y);
}
)";

    source = std::regex_replace(source, std::regex("texelFetch\\((\\w+)\\s*,\\s*ivec2\\(([^)]+)\\)\\s*,\\s*0\\)"),
                                "texelFetch($1, bufferCoords($2), 0)");

    size_t insertion_point = find_insertion_point(source);
    if (insertion_point != std::string::npos) {
        source.insert(insertion_point, boundaryProtection);
    }

    const char* uniformDecl = R"(
uniform int u_BufferTexWidth;
uniform int u_BufferTexHeight;
)";

    insertion_point = find_insertion_point(source);
    if (insertion_point != std::string::npos) {
        insertion_point = source.find('\n', insertion_point);
        if (insertion_point != std::string::npos) {
            source.insert(insertion_point + 1, uniformDecl);
        }
    }
}

static void inject_textureQueryLod(std::string& glsl) {
    const std::regex defRegex(R"(vec2\s+mg_textureQueryLod\s*\()", std::regex::ECMAScript);

    if (glsl.find("textureQueryLod") == std::string::npos) {
        return;
    }
    if (std::regex_search(glsl, defRegex)) {
        return;
    }

    const std::string textureQueryLodImpl = R"(
#define textureQueryLod mg_textureQueryLod

vec2 mg_textureQueryLod(sampler2D tex, vec2 uv) {
    vec2 texSizeF = vec2(textureSize(tex, 0));
    vec2 dFdx_uv = dFdx(uv * texSizeF);
    vec2 dFdy_uv = dFdy(uv * texSizeF);
    float maxDerivative = max(length(dFdx_uv), length(dFdy_uv));
    float lod = log2(maxDerivative);
    return vec2(lod);
}
)";

    size_t insertPos = find_insertion_point(glsl);
    glsl.insert(insertPos, "\n" + textureQueryLodImpl + "\n");
}

static inline void inject_temporal_filter(std::string& glsl) {
    const std::regex defRegex(R"(vec4\s+GI_TemporalFilter\s*\()", std::regex::ECMAScript);

    if (glsl.find("GI_TemporalFilter") == std::string::npos) {
        return;
    }
    if (std::regex_search(glsl, defRegex)) {
        return;
    }

    const std::regex uniformRegex(
        R"(^\s*(?:layout\s*\([^)]*\)\s*)?uniform\s+\w+(?:\s*\[\s*\d+\s*\])?\s+\w+(?:\s*\[\s*\d+\s*\])?\s*;.*$)",
        std::regex::ECMAScript | std::regex::multiline);
    std::sregex_iterator it(glsl.begin(), glsl.end(), uniformRegex);
    std::sregex_iterator end;
    size_t insertPos = 0;
    for (; it != end; ++it) {
        insertPos = it->position() + it->length();
    }

    const std::string GI_TemporalFilterImpl = R"(
vec4 GI_TemporalFilter() {
    vec2 uv = gl_FragCoord.xy / screenSize;
    uv += taaJitter * pixelSize;
    vec4 currentGI = texture(colortex0, uv);
    float depth = texture(depthtex0, uv).r;
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 viewPos = gbufferProjectionInverse * clipPos;
    viewPos /= viewPos.w;
    vec4 worldPos = gbufferModelViewInverse * viewPos;
    vec4 prevClipPos = gbufferPreviousProjection * (gbufferPreviousModelView * worldPos);
    prevClipPos /= prevClipPos.w;
    vec2 prevUV = prevClipPos.xy * 0.5 + 0.5;
    vec4 historyGI = texture(colortex1, prevUV);
    float difference = length(currentGI.rgb - historyGI.rgb);
    float thresholdValue = 0.1;
    float adaptiveBlend = mix(0.9, 0.0, smoothstep(thresholdValue, thresholdValue * 2.0, difference));
    vec4 filteredGI = mix(currentGI, historyGI, adaptiveBlend);
    if (difference > thresholdValue * 2.0) {
        filteredGI = currentGI;
    }
    return filteredGI;
}
)";
    glsl.insert(insertPos, "\n" + GI_TemporalFilterImpl + "\n");
}
#define xstr(s) str(s)
#define str(s) #s

void inject_mg_macro_definition(std::string& glslCode) {
    std::string macro_definitions =
        "\n#define MG_MOBILEGLUES\n"
        "#define MG_MOBILEGLUES_VERSION " xstr(MAJOR) xstr(MINOR) xstr(REVISION) xstr(PATCH) "\n";

    size_t versionPos = glslCode.rfind("#version");
    size_t insertionPos = 0;

    if (versionPos != std::string::npos) {
        size_t nextNewline = glslCode.find('\n', versionPos);
        insertionPos = (nextNewline != std::string::npos) ? nextNewline + 1 : glslCode.length();
    } else {
        size_t firstNewline = glslCode.find('\n');
        insertionPos = (firstNewline != std::string::npos) ? firstNewline + 1 : 0;
    }

    glslCode.insert(insertionPos, macro_definitions);
}

std::string preprocess_glsl(const std::string& glsl, GLenum shaderType) {
    std::string ret = glsl;
    // Remove lines beginning with `#line`
    ret = replace_line_starting_with(ret, "#line");
    // Act as if disable_GL_ARB_derivative_control is false
    replace_all(ret, "#ifdef GL_ARB_derivative_control", "#if 0");
    replace_all(ret, "#ifndef GL_ARB_derivative_control", "#if 1");

    // Polyfill transpose()
    replace_all(ret, "const mat3 rotInverse = transpose(rot);",
                "const mat3 rotInverse = mat3(rot[0][0], rot[1][0], rot[2][0], rot[0][1], rot[1][1], rot[2][1], "
                "rot[0][2], rot[1][2], rot[2][2]);");

    // GI_TemporalFilter injection
    inject_temporal_filter(ret);

    // textureQueryLod injection
    if (!g_gles_caps.GL_EXT_texture_query_lod) {
        inject_textureQueryLod(ret);
    }

    // MobileGlues macros injection
    inject_mg_macro_definition(ret);

    if (hardware->emulate_texture_buffer) {
        // Sampler buffer processing
        process_sampler_buffer(ret);
    }

    return ret;
}

int get_or_add_glsl_version(std::string& glsl) {
    int glsl_version = getGLSLVersion(glsl.c_str());
    if (glsl_version == -1) {
        glsl_version = 150;
        glsl.insert(0, "#version 150\n");
    } else if (glsl_version < 140) {
        // force upgrade glsl version
        glsl = replace_line_starting_with(glsl, "#version", "#version 150 compatibility\n");
        glsl_version = 150;
    }

    LOG_D("GLSL version: %d", glsl_version)
    return glsl_version;
}

std::vector<unsigned int> glsl_to_spirv(GLenum shader_type, int glsl_version, const char* const* shader_src,
                                        int& errc) {
    EShLanguage shader_language;
    switch (shader_type) {
    case GL_VERTEX_SHADER:
        shader_language = EShLanguage::EShLangVertex;
        break;
    case GL_FRAGMENT_SHADER:
        shader_language = EShLanguage::EShLangFragment;
        break;
    case GL_COMPUTE_SHADER:
        shader_language = EShLanguage::EShLangCompute;
        break;
    case GL_TESS_CONTROL_SHADER:
        shader_language = EShLanguage::EShLangTessControl;
        break;
    case GL_TESS_EVALUATION_SHADER:
        shader_language = EShLanguage::EShLangTessEvaluation;
        break;
    case GL_GEOMETRY_SHADER:
        shader_language = EShLanguage::EShLangGeometry;
        break;
    default:
        LOG_D("GLSL type not supported!")
        errc = -1;
        return {};
    }

    glslang::TShader shader(shader_language);
    shader.setStrings(shader_src, 1);

    using namespace glslang;
    shader.setEnvInput(EShSourceGlsl, shader_language, EShClientVulkan, glsl_version);
    shader.setEnvClient(EShClientOpenGL, EShTargetOpenGL_450);
    shader.setEnvTarget(EShTargetSpv, EShTargetSpv_1_5);
    shader.setAutoMapLocations(true);
    shader.setPreamble("#undef VULKAN\n");
    shader.setAutoMapBindings(true);

    TBuiltInResource TBuiltInResource_resources = InitResources();

    if (!shader.parse(&TBuiltInResource_resources, glsl_version, true, EShMsgDefault)) {
        LOG_D("GLSL Compiling ERROR: \n%s", shader.getInfoLog())
        errc = -1;
        return {};
    }
    LOG_D("GLSL Compiled.")

    glslang::TProgram program;
    program.addShader(&shader);

    if (!program.link(EShMsgDefault)) {
        LOG_D("Shader Linking ERROR: %s", program.getInfoLog())
        errc = -1;
        return {};
    }
    LOG_D("Shader Linked.")
    std::vector<unsigned int> spirv_code;
    glslang::SpvOptions spvOptions;
    spvOptions.disableOptimizer = false;
    glslang::GlslangToSpv(*program.getIntermediate(shader_language), spirv_code, &spvOptions);
    errc = 0;
    return spirv_code;
}

// The context owns the ParsedIR, the compiler and every string they hand back, and the only
// destroy used to sit past the early return. A shader the ES backend rejects is a normal
// outcome and failed translations are not cached, so that leaked the lot again on every
// resource-pack reload. Scoped so no exit can skip it.
namespace {
struct spvc_context_guard_t {
    spvc_context context = nullptr;
    spvc_context_guard_t() = default;
    ~spvc_context_guard_t() {
        if (context) spvc_context_destroy(context);
    }
    spvc_context_guard_t(const spvc_context_guard_t&) = delete;
    spvc_context_guard_t& operator=(const spvc_context_guard_t&) = delete;
};
} // namespace

// SPIRV-Cross throws internally and turns that into a result code at its C boundary; on failure
// it leaves the out-parameter untouched. Dropping the code therefore hands the next call a
// handle that was never written, which crashes rather than reporting anything.
static bool spvc_ok(spvc_context context, spvc_result res, const char* what) {
    if (res == SPVC_SUCCESS) {
        return true;
    }
    LOG_E("Error: %s failed in spirv-cross: %s", what, spvc_context_get_last_error_string(context))
    return false;
}

std::string spirv_to_essl(std::vector<unsigned int> spirv, uint essl_version, int& errc) {
    spvc_parsed_ir ir = nullptr;
    spvc_compiler compiler_glsl = nullptr;
    spvc_compiler_options options = nullptr;
    const char* result = nullptr;

    const SpvId* p_spirv = spirv.data();
    size_t word_count = spirv.size();

    LOG_D("spirv_code.size(): %d", spirv.size())

    // Declared before 'essl': the compiled source lives in context-owned memory and is only
    // copied out when the std::string is constructed, so the guard has to outlive it.
    spvc_context_guard_t guard;
    if (spvc_context_create(&guard.context) != SPVC_SUCCESS || !guard.context) {
        LOG_E("Error: could not create a spirv-cross context.")
        errc = -1;
        return "";
    }
    spvc_context context = guard.context;

    if (!spvc_ok(context, spvc_context_parse_spirv(context, p_spirv, word_count, &ir), "spvc_context_parse_spirv") ||
        !ir) {
        errc = -1;
        return "";
    }
    if (!spvc_ok(context,
                 spvc_context_create_compiler(context, SPVC_BACKEND_GLSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                                              &compiler_glsl),
                 "spvc_context_create_compiler") ||
        !compiler_glsl) {
        errc = -1;
        return "";
    }
    if (!spvc_ok(context, spvc_compiler_create_compiler_options(compiler_glsl, &options),
                 "spvc_compiler_create_compiler_options") ||
        !options) {
        errc = -1;
        return "";
    }
    // A silently dropped GLSL_ES option would emit desktop GLSL and hand it straight to the
    // driver, so these are checked too.
    if (!spvc_ok(context,
                 spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_GLSL_VERSION,
                                                essl_version >= 300 ? essl_version : 300),
                 "spvc_compiler_options_set_uint") ||
        !spvc_ok(context, spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_GLSL_ES, SPVC_TRUE),
                 "spvc_compiler_options_set_bool") ||
        !spvc_ok(context, spvc_compiler_install_compiler_options(compiler_glsl, options),
                 "spvc_compiler_install_compiler_options")) {
        errc = -1;
        return "";
    }
    if (!spvc_ok(context, spvc_compiler_compile(compiler_glsl, &result), "spvc_compiler_compile") || !result) {
        errc = -1;
        return "";
    }

    std::string essl = result;

    errc = 0;
    return essl;
}

static bool glslang_inited = false;
std::string GLSLtoGLSLES_2(const char* glsl_code, GLenum glsl_type, uint essl_version, int& return_code) {
    std::string correct_glsl_str = preprocess_glsl(glsl_code, glsl_type);
    LOG_D("Firstly converted GLSL:\n%s", correct_glsl_str.c_str())
    int glsl_version = get_or_add_glsl_version(correct_glsl_str);

    if (!glslang_inited) {
        glslang::InitializeProcess();
        glslang_inited = true;
    }
    const char* s[] = {correct_glsl_str.c_str()};
    int errc = 0;
    std::vector<unsigned int> spirv_code = glsl_to_spirv(glsl_type, glsl_version, s, errc);
    if (errc != 0) {
        return_code = -1;
        return "";
    }
    errc = 0;
    std::string essl = spirv_to_essl(spirv_code, essl_version, errc);
    if (errc != 0) {
        return_code = -2;
        return "";
    }

    // Post-processing ESSL

    if (glsl_type != GL_COMPUTE_SHADER) {
        essl = removeLayoutBinding(essl);
    }
    essl = processOutColorLocations(essl);
    essl = forceSupporterOutput(essl);

    LOG_D("Originally GLSL to GLSL ES Complete: \n%s", essl.c_str())
    return_code = errc;
    return essl;
}

std::string GLSLtoGLSLES_1(const char* glsl_code, GLenum glsl_type, uint esversion, int& return_code) { // useless now
    /*
#if !defined(__APPLE__)
    LOG_W("Warning: use glsl optimizer to convert shader.")
    if (esversion < 300) esversion = 300;
    std::string result = MesaConvertShader(glsl_code, glsl_type == GL_VERTEX_SHADER ? GL_VERTEX_SHADER :
GL_FRAGMENT_SHADER, 460LL, esversion);

    return_code = 0;
    return result;
#else
    LOG_W_FORCE("Cannot convert glsl with version %d in MacOS/iOS", esversion);
    return std::string(glsl_code);
#endif
    */
}
