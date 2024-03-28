DOCKER_REGISTRY ?= docker.io

python_version=3.12
user=alezana
image_name=dg_base
tagname=$(python_version)
dockerfile=Dockerfile.base

tag=$(DOCKER_REGISTRY)/$(user)/$(image_name):$(tagname)
platforms=linux/amd64,linux/arm64

# Build and push the image to the registry
build:
	docker buildx build $(BUILDX_EXTRA_OPTS) --platform $(platforms) -f $(dockerfile) -t $(tag) --build-arg PYTHON_VERSION_=${python_version}  --push .

build-no-cache:
	$(MAKE) build BUILDX_EXTRA_OPTS=--no-cache

push-on-dockerhub:
	docker login
	docker push $(tag)
