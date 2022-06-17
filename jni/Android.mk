LOCAL_PATH:=$(call my-dir)

include $(CLEAR_VARS)

LOCAL_CFLAGS += -fPIE -mfloat-abi=softfp -mfpu=neon -fopenmp -g
LOCAL_LDFLAGS += -fPIE -fopenmp -static-openmp # -pie

LOCAL_MODULE:=project_basis

LOCAL_SRC_FILES:=recognition_seq.c main.c

LOCAL_LDLIBS := 

LOCAL_CFLAGS += 

include $(BUILD_EXECUTABLE)
