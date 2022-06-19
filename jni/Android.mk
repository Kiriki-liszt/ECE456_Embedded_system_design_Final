LOCAL_PATH:=$(call my-dir)

include $(CLEAR_VARS)

LOCAL_CFLAGS += -fPIE -mfloat-abi=soft -g # -mfloat-abi=softfp   //fp -mfpu=neon
LOCAL_LDFLAGS += -fPIE # -pie

LOCAL_MODULE:=recognition_seq

LOCAL_SRC_FILES:=recognition_seq.c main.c

LOCAL_LDLIBS := 

LOCAL_CFLAGS += 

include $(BUILD_EXECUTABLE)
