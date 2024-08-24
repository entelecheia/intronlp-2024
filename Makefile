# To do stuff with make, you type `make` in a directory that has a file called
# "Makefile".  You can also type `make -f <makefile>` to use a different filename.
#
# A Makefile is a collection of rules. Each rule is a recipe to do a specific
# thing, sort of like a grunt task or an npm package.json script.
#
# A rule looks like this:
#
# <target>: <prerequisites...>
# 	<commands>
#
# The "target" is required. The prerequisites are optional, and the commands
# are also optional, but you have to have one or the other.
#
# Type `make` to show the available targets and a description of each.
#
.DEFAULT_GOAL := help
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


##@ Clean-up

clean: ## run all clean commands
	@poe clean

##@ Utilities

large-files: ## show the 20 largest files in the repo
	@find . -printf '%s %p\n'| sort -nr | head -20

disk-usage: ## show the disk usage of the repo
	@du -h -d 2 .

git-sizer: ## run git-sizer
	@git-sizer --verbose

gc-prune: ## garbage collect and prune
	@git gc --prune=now

##@ Setup

install-node: ## install node
	@export NVM_DIR="$${HOME}/.nvm"; \
	[ -s "$${NVM_DIR}/nvm.sh" ] && . "$${NVM_DIR}/nvm.sh"; \
	nvm install --lts

nvm-ls: ## list node versions
	@export NVM_DIR="$${HOME}/.nvm"; \
	[ -s "$${NVM_DIR}/nvm.sh" ] && . "$${NVM_DIR}/nvm.sh"; \
	nvm ls

set-default-node: ## set default node
	@export NVM_DIR="$${HOME}/.nvm"; \
	[ -s "$${NVM_DIR}/nvm.sh" ] && . "$${NVM_DIR}/nvm.sh"; \
	nvm alias default node

install-pipx: ## install pipx (pre-requisite for external tools)
	@command -v pipx &> /dev/null || pip install --user pipx || true

install-copier: install-pipx ## install copier (pre-requisite for init-project)
	@command -v copier &> /dev/null || pipx install copier || true

install-poetry: install-pipx ## install poetry (pre-requisite for install)
	@command -v poetry &> /dev/null || pipx install poetry || true

install-poe: install-pipx ## install poetry (pre-requisite for install)
	@command -v poe &> /dev/null || pipx install poethepoet || true

install-commitzen: install-pipx ## install commitzen (pre-requisite for commit)
	@command -v cz &> /dev/null || pipx install commitizen || true

install-precommit: install-pipx ## install pre-commit
	@command -v pre-commit &> /dev/null || pipx install pre-commit || true

install-precommit-hooks: install-precommit ## install pre-commit hooks
	@pre-commit install

mkvirtualenv: ## create the project environment
	@python3 -m venv "$$WORKON_HOME/deepnlp-2023"
	@. "$$WORKON_HOME/deepnlp-2023/bin/activate"
	@pip install --upgrade pip setuptools wheel

mkvirtualenv-system: ## create the project environment with system site packages
	@python3 -m venv "$$WORKON_HOME/deepnlp-2023" --system-site-packages
	@. "$$WORKON_HOME/deepnlp-2023/bin/activate"
	@pip install --upgrade pip setuptools wheel

workon: ## activate the project environment
	@. "$$WORKON_HOME/deepnlp-2023/bin/activate"

initialize: install-pipx ## initialize the project environment
	@pipx install copier
	@pipx install poethepoet
	@pipx install commitizen
	@pipx install pre-commit
	@pre-commit install

remove-template: ## remove template-specific files
	@rm -rf src/nlp2024
	@rm -rf tests/nlp2024
	@rm -rf CHANGELOG.md
	@echo "Template-specific files removed."

init-project: initialize remove-template ## initialize the project (Warning: do this only once!)
	@copier copy --trust --answers-file .copier-config.yaml gh:entelecheia/hyperfast-course-template .

reinit-project: install-copier ## reinitialize the project (Warning: this may overwrite existing files!)
	@bash -c 'args=(); while IFS= read -r file; do args+=("--skip" "$$file"); done < .copierignore; copier copy --trust "$${args[@]}" --answers-file .copier-config.yaml gh:entelecheia/hyperfast-course-template .'

reinit-docker-project: install-copier ## reinitialize the project (Warning: this may overwrite existing files!)
	@bash -c 'args=(); while IFS= read -r file; do args+=("--skip" "$$file"); done < .copierignore; copier copy "$${args[@]}" --answers-file .copier-docker-config.yaml --trust gh:entelecheia/hyperfast-docker-template .'

##@ Docker

symlink-global-docker-env: ## symlink global docker env file for local development
	@DOCKERFILES_SHARE_DIR="$HOME/.local/share/dockerfiles" \
	DOCKER_GLOBAL_ENV_FILENAME=".env.docker" \
	DOCKER_GLOBAL_ENV_FILE="$${DOCKERFILES_SHARE_DIR}/$${DOCKER_GLOBAL_ENV_FILENAME}" \
	[ -f "$${DOCKER_GLOBAL_ENV_FILE}" ] && ln -sf "$${DOCKER_GLOBAL_ENV_FILE}" .env.docker || echo "Global docker env file not found"

docker-login: ## login to docker
	@bash .docker/.docker-scripts/docker-compose.sh login

docker-build: ## build the docker app image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh build

docker-config: ## show the docker app config
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh config

docker-push: ## push the docker app image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh push

docker-run: ## run the docker base image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh run

docker-up: ## launch the docker app image
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh up

docker-up-detach: ## launch the docker app image in detached mode
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh up --detach

docker-tag-latest: ## tag the docker app image as latest
	@IMAGE_VARIANT=$${IMAGE_VARIANT:-"base"} \
	DOCKER_PROJECT_ID=$${DOCKER_PROJECT_ID:-"default"} \
	bash .docker/.docker-scripts/docker-compose.sh tag
