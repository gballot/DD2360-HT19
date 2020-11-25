# automatic compilation and run for tegner or lab
STATION=tegner

EXEC:=exercise_3.out
OBJ:=exercise_3.o

PARAM:=-s 1024 -v

ifeq (${STATION}, tegner)
	ARCH:=sm_30
	RUN:=srun -n 1
else ifeq (${STATION}, lab)
	ARCH:=sm_50
else
	ARCH:=sm_37
endif

NVCC:=nvcc
NVFLAGS:=-O3 -arch=${ARCH}
LDFLAGS:=-lcurand -lcublas


all: ${EXEC}

${EXEC}: ${OBJ}
	${NVCC} ${NVFLAGS} -o $@ $< ${LDFLAGS}

%.o: %.cu
	${NVCC} ${NVFLAGS} -o $@ -c $< ${LDFLAGS}

run: ${EXEC}
	${RUN} ./${EXEC} ${PARAM}

compare: ${EXEC}
	sbatch ./run.sh

alloc:
	salloc -A edu20.dd2360 -C Haswell -t 0:05:00 --gres=gpu:K420:1 --nodes=1

tags:
	ctags --langmap=c++:+.cu *.cu

clean:
	${RM} *.o ${EXEC}

.PHONY: all clean alloc run tags
