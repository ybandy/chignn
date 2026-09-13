SPDK_ROOT_DIR := $(abspath $(CURDIR)/../spdk)

CXX = g++

CXXFLAGS = -O3 -g -march=native
CXXFLAGS += -D_GNU_SOURCE -fPIC -pthread
CXXFLAGS += -I$(SPDK_ROOT_DIR)/include -I.

LDFLAGS = -L../spdk_lib
LDFLAGS += -Wl,--whole-archive -Wl,-Bstatic -lspdk
LDFLAGS += -Wl,--no-whole-archive -Wl,-Bstatic -ldpdk
LDFLAGS += $(SPDK_ROOT_DIR)/isa-l/.libs/libisal.a
LDFLAGS += -Wl,-Bdynamic
LDFLAGS += -lnuma
LDFLAGS += -luuid
LDFLAGS += -lssl
LDFLAGS += -lcrypto
LDFLAGS += -lpthread -lm


TARGET = write_npy_to_disk

SRCS = $(TARGET).cc spdk_util.cc npy_reader.cc
OBJS = $(SRCS:.cc=.o) 
DEPS = $(SRCS:.cc=.d) 


all: $(TARGET)

$(TARGET): $(OBJS)
	g++ $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)
