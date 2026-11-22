// include/strata/gemma4/engine.hpp - the Gemma 4 26B A4B (GGUF architecture gemma4) engine path as a library.
// strata-g4 is a thin wrapper around this entry point.  See src/gemma4/g4_engine.cu.
#pragma once

namespace strata::gemma4 {

/// The engine's command line (see the usage text in g4_engine.cu).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::gemma4
