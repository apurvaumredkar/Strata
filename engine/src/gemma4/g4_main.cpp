// src/gemma4/g4_main.cpp - strata-g4: the Gemma 4 26B A4B engine path on its own.
#include "strata/gemma4/engine.hpp"

#include <cstdlib>
#include <initializer_list>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

// Windows may run a process it takes for background work at a lower clock speed and with a coarse timer ("power
// throttling", EcoQoS), and a laptop does so readily.  The CPU pool computes the experts the card does not hold, so
// the engine asks to be left out.  G4_POWER_THROTTLING=keep leaves the system's choice, for comparisons.
void no_power_throttling() {
#ifdef _WIN32
    const char* e = std::getenv("G4_POWER_THROTTLING");
    if (e && e[0] == 'k') return;
    for (const ULONG mask : {0x1ul | 0x4ul, 0x1ul}) {   // execution speed and timer resolution; Windows 10 knows the first
        PROCESS_POWER_THROTTLING_STATE off{PROCESS_POWER_THROTTLING_CURRENT_VERSION, mask, 0};
        if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &off, sizeof(off))) return;
    }
#endif
}

}  // namespace

int main(int argc, char** argv) {
    no_power_throttling();
    return strata::gemma4::run(argc, argv);
}
