LOCAL_PATH:=$(call my-dir)

include $(CLEAR_VARS)

LOCAL_CFLAGS += -fPIE
LOCAL_LDFLAGS += -fPIE # -pie

LOCAL_MODULE:=recognition_seq

LOCAL_SRC_FILES:=srecognition_seq.c main.c

LOCAL_LDLIBS := 

LOCAL_CFLAGS += 

include $(BUILD_EXECUTABLE)
