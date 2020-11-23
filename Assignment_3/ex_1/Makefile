# automatic compilation and run for tegner or lab
STATION=tegner

EXEC:=hw3_ex1.out
OBJ:=hw3_ex1.o

PARAM:=images/hw3.bmp
RES:=images/hw3_result_1.bmp images/hw3_result_2.bmp images/hw3_result_3.bmp

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

${RES}: ${EXEC}
	${MAKE} run

res1: ${RES}
	display -resize 1280x720 images/hw3_result_1.bmp

res2: ${RES}
	montage -tile 2x1 -crop 320x180+512+512 -geometry 640x360 images/hw3_result_1.bmp images/hw3_result_2.bmp images/hw3_result_2_comp.jpg
	display images/hw3_result_2_comp.jpg

res3: ${RES}
	montage -border 0 -geometry 640x360 -tile 3x1 images/hw3.bmp images/hw3_result_1.bmp images/hw3_result_3.bmp images/hw3_result_3_comp.jpg
	display images/hw3_result_3_comp.jpg

clean:
	${RM} *.o ${EXEC} ${RES} images/hw3_result_3_comp.jpg images/hw3_result_2_comp.jpg

.PHONY: all clean alloc run tags res1 res2 res3
