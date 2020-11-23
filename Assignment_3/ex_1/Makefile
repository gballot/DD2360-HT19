# automatic compilation and run for tegner or lab
STATION=tegner

EXEC:=hw3_ex1.out
OBJ:=hw3_ex1.o

PARAM:=images/hw3.bmp

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


all: ${EXEC}

${EXEC}: ${OBJ}
	${NVCC} ${NVFLAGS} -o $@ $<

%.o: %.cu
	${NVCC} ${NVFLAGS} -o $@ -c $<

run: ${EXEC}
	${RUN} ./${EXEC} ${PARAM}

alloc:
	salloc -A edu20.dd2360 -C Haswell -t 0:05:00 --gres=gpu:K420:1 --nodes=1

tags:
	ctags --langmap=c++:+.cu *.cu

display:
	display -resize 1280x720 images/hw3_result_1.bmp

clean:
	${RM} *.o ${EXEC}

.PHONY: all clean alloc run tags display
