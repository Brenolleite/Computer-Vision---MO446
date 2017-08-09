DOCKER_IMAGE = adnrv/opencv

SOURCE_DIR = /home/breno/git/Computer_Vision-MO446/p0-XX-YY/
DOCKER_DIR = /builds

DOCKER_DISPLAY_ARGS = --env="DISPLAY" --net=host --env="QT_X11_NO_MITSHM=1" \
                      --volume="${HOME}/.Xauthority:/root/.Xauthority:rw" \

DOCKER_VOLUMES = --volume=$(SOURCE_DIR):$(DOCKER_DIR) --workdir=$(DOCKER_DIR)

DOCKER_RUN = docker run -it --rm  $(DOCKER_RUN_ARGS) $(DOCKER_VOLUMES) $(DOCKER_IMAGE)
DOCKER_RUN_DISPLAY = docker run -it --rm  $(DOCKER_RUN_ARGS) $(DOCKER_DISPLAY_ARGS) $(DOCKER_VOLUMES) $(DOCKER_IMAGE)

all: exec

exec:
	$(DOCKER_RUN_DISPLAY) bash -c "make"; \
	status=$$?

.PHONY: all exec
