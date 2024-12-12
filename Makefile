CFLAGS = -std=c++20 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

.PHONY: test clean

test: main
	./main

clean:
	rm main

main: main.cpp
	clang++ $(CFLAGS) -o main main.cpp $(LDFLAGS)
