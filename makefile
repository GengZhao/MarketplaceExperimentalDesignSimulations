CXX=g++
OPTFLAG=-O3
CXXFLAGS=-Wall -g $(OPTFLAG) -std=c++14

EXECUTABLES=main eval_estimators test

all: $(EXECUTABLES)

main: main.o Marketplace.o
	$(CXX) $(CXXFLAGS) -o main main.o Marketplace.o

test: test.o
	$(CXX) $(CXXFLAGS) -o test test.o

eval_estimators: eval_estimators.o Marketplace.o
	$(CXX) $(CXXFLAGS) -pthread -o eval_estimators eval_estimators.o Marketplace.o

main.o: main.cc Marketplace.h
	$(CXX) $(CXXFLAGS) -c main.cc

test.o: test.cc
	$(CXX) $(CXXFLAGS) -c test.cc

eval_estimators.o: eval_estimators.cc Marketplace.h utils.h
	$(CXX) $(CXXFLAGS) -c eval_estimators.cc

Marketplace.o: Marketplace.cc Marketplace.h
	$(CXX) $(CXXFLAGS) -c Marketplace.cc

clean:
	$(RM) $(EXECUTABLES) *.o *~

