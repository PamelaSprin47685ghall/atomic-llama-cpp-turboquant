if (NOT DEFINED WANXIANGQI_REPETITION_ROOT OR
    NOT DEFINED WANXIANGQI_REPETITION_MANIFEST OR
    NOT DEFINED WANXIANGQI_REPETITION_OUTPUT OR
    NOT DEFINED WANXIANGQI_REPETITION_TOOL)
    message(FATAL_ERROR "missing repetition-envelope update arguments")
endif()

find_program(WANXIANGQI_GIT_EXECUTABLE git REQUIRED)
execute_process(
    COMMAND "${WANXIANGQI_GIT_EXECUTABLE}" -C "${WANXIANGQI_REPETITION_ROOT}" ls-files --cached
    OUTPUT_VARIABLE corpus
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE git_result)
if (NOT git_result EQUAL 0)
    message(FATAL_ERROR "failed to enumerate repository corpus")
endif()

file(WRITE "${WANXIANGQI_REPETITION_MANIFEST}" "${corpus}\n")
execute_process(
    COMMAND "${WANXIANGQI_REPETITION_TOOL}"
        "${WANXIANGQI_REPETITION_ROOT}"
        "${WANXIANGQI_REPETITION_MANIFEST}"
        "${WANXIANGQI_REPETITION_OUTPUT}"
    RESULT_VARIABLE update_result)
if (NOT update_result EQUAL 0)
    message(FATAL_ERROR "repetition-envelope update failed: ${update_result}")
endif()
