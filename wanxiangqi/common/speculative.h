#pragma once

#include "../../common/speculative.h"

// Fork extension used by calibration tools that execute the MTP
// draft graph across independent branches/chunks. Keep this out of the
// upstream public speculative header.
void common_speculative_reset(common_speculative * spec);
