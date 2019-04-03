CC = CC
MPCC = CC
OPENMP = -openmp
CFLAGS = -O3 -std=c++11
LIBS =


TARGETS = serial openmp mpi autograder

all:	$(TARGETS)

autograder: autograder.o common.o matrixCells.o
	$(CC) -o $@ $(LIBS) autograder.o common.o matrixCells.o
mpi: mpi.o common.o matrixCells.o
	$(CC) -o $@ $(LIBS) mpi.o common.o matrixCells.o
serial: serial.o common_part1.o matrixCells_part1.o
	$(CC) -o $@ $(LIBS) serial.o common_part1.o matrixCells_part1.o
openmp: openmp.o common_part1.o matrixCells_part1.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o common_part1.o matrixCells_part1.o


autograder.o: autograder.cpp common.h matrixCells.h
	$(CC) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common_part1.h matrixCells_part1.cpp 
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common_part1.h matrixCells_part1.cpp 
	$(CC) -c $(CFLAGS) serial.cpp
mpi.o: mpi.cpp common.h
	$(CC) -c $(CFLAGS) mpi.cpp
matrixCells.o: common.h matrixCells.h matrixCells.cpp
	$(CC) -c $(CFLAGS) matrixCells.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp
common_part1.o: common_part1.cpp common_part1.h
	$(CC) -c $(CFLAGS) common_part1.cpp
matrixCells_part1.o: common_part1.h matrixCells_part1.h matrixCells_part1.cpp
	$(CC) -c $(CFLAGS) matrixCells_part1.cpp

clean: