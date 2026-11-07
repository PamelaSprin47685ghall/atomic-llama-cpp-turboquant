#pragma once

// TP5 host-pool feature switches.
//
// Every switch is off unless the environment value is exactly "1". The
// default path is the pre-pool implementation, so an unset switch keeps the
// old allocation and ordering behavior.

#include <cstdlib>

namespace tp5 {

inline bool flag_exact_one(const char * name) {
    const char * value = std::getenv(name);

    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

} // namespace tp5
