FLAGS=`pkg-config --cflags --libs opencv`
all: mrc.cpp main.cpp
	g++ $(FLAGS)  -std=c++11 mrc.cpp main.cpp 
clean: 
	rm -rf ./*.o ./*.out
