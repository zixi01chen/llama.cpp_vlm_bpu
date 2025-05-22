# llava (legacy)
add_library(llava OBJECT
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            )

target_link_libraries(llava PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(llava PUBLIC .)
target_include_directories(llava PUBLIC ../..)
target_include_directories(llava PUBLIC ../../common)

target_compile_features(llava PRIVATE cxx_std_17)

add_library(llava_static STATIC $<TARGET_OBJECTS:llava>)
if (BUILD_SHARED_LIBS)
    set_target_properties(llava PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llava PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(llava_shared SHARED $<TARGET_OBJECTS:llava>)
    target_link_libraries(llava_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS llava_shared LIBRARY)
endif()

# mtmd

add_library(mtmd OBJECT
            mtmd.cpp
            mtmd-helper.cpp
            mtmd.h
            clip.cpp
            clip.h
            clip-impl.h
            )

target_link_libraries(mtmd PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(mtmd PUBLIC .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../common) # for stb_image.h

target_compile_features(mtmd PRIVATE cxx_std_17)

add_library(mtmd_static STATIC $<TARGET_OBJECTS:mtmd>)
if (BUILD_SHARED_LIBS)
    set_target_properties(mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(mtmd_shared SHARED $<TARGET_OBJECTS:mtmd>)
    target_link_libraries(mtmd_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS mtmd_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(llava PRIVATE -Wno-cast-qual) # stb_image.h
    target_compile_options(mtmd PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(llava BUILD_INFO)
    add_dependencies(mtmd BUILD_INFO)
endif()

add_executable(llama-llava-cli    deprecation-warning.cpp)
add_executable(llama-gemma3-cli   deprecation-warning.cpp)
add_executable(llama-minicpmv-cli deprecation-warning.cpp)
add_executable(llama-qwen2vl-cli  deprecation-warning.cpp)

set(TARGET llama-mtmd-cli)
add_executable(${TARGET} mtmd-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-mtmd-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-llava-clip-quantize-cli)
add_executable(${TARGET} clip-quantize-cli.cpp)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME llama-llava-clip-quantize-cli)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)


find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc highgui imgcodecs)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 4 (core imgproc highgui imgcodecs) not found!")
endif()

# ----------------------------------------
# intern2vl-bpu-cli 可执行文件
# ----------------------------------------
set(TARGET llama-intern2vl-bpu-cli)

# 1. 定义可执行文件及其源文件
add_executable(${TARGET}
    intern2vl-bpu-cli.cpp   # intern2vl-bpu 命令行入口
)

# 2. 指定最终生成的可执行文件名称（去除前缀）
set_target_properties(${TARGET} PROPERTIES
    OUTPUT_NAME "llama-intern2vl-bpu-cli"
)

# 3. 指定安装规则，将可执行文件放到 bin 目录
install(TARGETS ${TARGET}
    RUNTIME DESTINATION bin
)
link_directories("/usr/hobot/lib")
link_directories("/usr/lib")
link_directories("/lib")
# 4. 包含头文件目录 —— PRIVATE 表示仅用于本 target
target_include_directories(${TARGET} PRIVATE
    .                       # 当前目录，包含 intern2vl-bpu-cli.h 等
    ../..                   # 项目根目录，包含 llama.cpp/ggml 头文件
    ../../common            # common 库头文件
    ${OpenCV_INCLUDE_DIRS}  # OpenCV 查找到的头文件路径
    ${SYS_ROOT}/usr/include/hobot  # Hobot BPU SDK 头文件（确保 SYS_ROOT 已设置）
)

# 5. 编译选项 —— 使用 C++17 标准
target_compile_features(${TARGET} PRIVATE
    cxx_std_17
)

# 6. 链接依赖库 —— PRIVATE 表示仅影响本 target
target_link_libraries(${TARGET} PRIVATE
    common                  # llama.cpp 的 common 库
    llava_static            # llava 静态库（也可改为 llava_shared）
    ${OpenCV_LIBS}          # OpenCV 库
    ${CMAKE_THREAD_LIBS_INIT}  # 线程库
    # Hobot BPU 动态/静态库
    dnn
    /usr/hobot/lib/libalog.so
    /usr/hobot/lib/libcnn_intf.so
)
