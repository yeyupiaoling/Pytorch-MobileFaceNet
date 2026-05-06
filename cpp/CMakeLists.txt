cmake_minimum_required(VERSION 3.18)
project(face_recognition_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# 复制运行时 DLL 到可执行文件目录，避免运行时缺少依赖
function(copy_runtime_dlls target_name)
    if(NOT WIN32)
        return()
    endif()

    # 复制 LibTorch 运行时 DLL
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    if(TORCH_DLLS)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
        )
    endif()

    # 复制 OpenCV 运行时 DLL，兼容不同安装路径
    set(OPENCV_RUNTIME_DIR_CANDIDATES
        "${OpenCV_DIR}/../bin"
        "${OpenCV_DIR}/../../bin"
        "${OpenCV_DIR}/../../../bin"
        "${OpenCV_DIR}/x64/vc16/bin"
        "${OpenCV_DIR}/x64/vc17/bin"
    )

    foreach(candidate_dir IN LISTS OPENCV_RUNTIME_DIR_CANDIDATES)
        if(EXISTS "${candidate_dir}")
            file(GLOB OPENCV_DLLS "${candidate_dir}/*.dll")
            if(OPENCV_DLLS)
                add_custom_command(TARGET ${target_name} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${OPENCV_DLLS}
                        $<TARGET_FILE_DIR:${target_name}>
                    COMMAND_EXPAND_LISTS
                )
                break()
            endif()
        endif()
    endforeach()
endfunction()

# ========== 人脸检测库 ==========
add_library(mtcnn_detector
    src/mtcnn_detector.cpp
)

target_include_directories(mtcnn_detector
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mtcnn_detector
    PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)

# ========== 人脸识别库 ==========
add_library(face_recognizer
    src/face_recognizer.cpp
)

target_include_directories(face_recognizer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(face_recognizer
    PUBLIC
        mtcnn_detector
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)

# ========== 图片识别示例 ==========
add_executable(face_image_demo
    examples/image_demo.cpp
)

target_link_libraries(face_image_demo
    PRIVATE
        face_recognizer
)
copy_runtime_dlls(face_image_demo)

# ========== 摄像头识别示例 ==========
add_executable(face_camera_demo
    examples/camera_demo.cpp
)

target_link_libraries(face_camera_demo
    PRIVATE
        face_recognizer
)
copy_runtime_dlls(face_camera_demo)

# MSVC 编译选项：启用 UTF-8 源码编码
if(MSVC)
    target_compile_options(mtcnn_detector PRIVATE /utf-8)
    target_compile_options(face_recognizer PRIVATE /utf-8)
    target_compile_options(face_image_demo PRIVATE /utf-8)
    target_compile_options(face_camera_demo PRIVATE /utf-8)
endif()
