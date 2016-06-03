FLAGS=`pkg-config --cflags --libs opencv`
split: mrc.cpp split.cpp
	g++ $(FLAGS)  -std=c++11 mrc.cpp split.cpp -o split
scan: mrc.cpp scan.cpp
	g++ $(FLAGS)  -std=c++11 mrc.cpp scan.cpp -o scan
clean: 
	rm -rf  split scan
