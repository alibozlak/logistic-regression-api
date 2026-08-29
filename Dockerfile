# syntax=docker/dockerfile:1

# ---- build stage ----
# Oracle GraalVM for JDK 25, with `native-image` already installed. Maven itself
# comes from the wrapper in this tree, so the build is pinned by the repository
# rather than by whatever version the image happens to ship.
#
# JDK 25 and not the 21 this project targets, because GraalVM for JDK 21 cannot
# build a Spring Boot 4 application at all: its Substrate VM is the 23.1 line,
# which does not understand the `typeReached` conditions in the reachability
# metadata and dies in SerializationFeature with a NoClassDefFoundError. The
# bytecode is still `--release 21` — only the toolchain moved.
#
# The `-ol9` variant is deliberate. The default tag sits on Oracle Linux 10 and
# links against its glibc 2.39, which no Debian stable before trixie can satisfy;
# Oracle Linux 9 links against 2.34 instead, and glibc is forward compatible, so
# a binary built here is met by the runtime base below and by anything newer.
FROM container-registry.oracle.com/graalvm/native-image:25-ol9 AS builder

WORKDIR /app

# Pull the dependency graph on a layer keyed only on the POM, so that editing
# src/ costs a compile and not another walk of Maven Central. The wrapper's own
# download of Maven lands on this layer too. Both are ordinary image layers
# rather than BuildKit cache mounts, because the daemon this is built against
# runs the legacy builder, which has no idea what a cache mount is.
#
# The image ships curl but no unzip; the wrapper notices and fetches the .tar.gz
# distribution instead of the .zip on its own.
COPY .mvn/ .mvn/
COPY mvnw pom.xml ./
RUN ./mvnw -B -Pnative dependency:go-offline

COPY src/ src/

# `native:compile` forks a `package` run of its own first, which is what gets
# Spring's AOT step (`process-aot`, contributed by the parent POM's `native`
# profile) to write out the reflection metadata before native-image starts.
# Running `package` alone would produce an ordinary jar and nothing else: the
# parent declares native-maven-plugin under pluginManagement only, so no
# lifecycle phase is bound to the image build.
#
# This is the expensive step: about two minutes on 16 cores, and 6.3 GB of heap —
# native-image takes 75% of whatever memory the container can see. A build that dies
# here with no error message ran out of it rather than failing to compile; the lever
# is the WSL VM's allocation in .wslconfig. See the README.
RUN ./mvnw -B -Pnative -DskipTests native:compile

# ---- runtime stage ----
# Nothing of the JDK survives into this stage — that is the point of compiling
# ahead of time. What the binary still needs from the outside is glibc and zlib,
# both of which the slim Debian base already carries.
FROM debian:bookworm-slim AS runtime

# The server reads no files and binds an unprivileged port, so it has no reason
# to run as root.
RUN useradd --system --user-group --no-create-home app

COPY --from=builder /app/target/logistic-regression-api /usr/local/bin/logistic-regression-api

USER app

# Spring Boot listens on every interface unless `server.address` says otherwise,
# so — unlike the Rust services in this chain — there is nothing to override here
# to make the container reachable. Keep it private by publishing to `127.0.0.1`
# on the host side, or by not publishing the port at all.
#
# The port is the framework's own default, left alone so a container and a local
# `./mvnw spring-boot:run` answer in the same place. Move it with
# `-e SERVER_PORT=...`, or more simply by remapping: `-p 127.0.0.1:3002:8080`.
EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/logistic-regression-api"]
