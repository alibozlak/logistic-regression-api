# logistic-regression-api

Binary logistic regression trained with batch gradient descent, exposed over HTTP.
Written from scratch — no ML library, no autodiff — so that every step of the math
is visible in the source.

The service keeps no state. `/train` returns the coefficients it converged to and
forgets them; `/predict` is handed a set of coefficients along with the sample to
score. Between the two sits one wrinkle worth reading before trusting a round trip —
see [Known limitations](#known-limitations).

It compiles ahead of time as well as to bytecode: `./mvnw -Pnative native:compile`
produces a standalone executable, and the `Dockerfile` runs that build itself, so an
image can be produced with no JDK on the host at all.

## The math

The model puts a plain linear combination of the `n` features through a sigmoid:

```text
z(x) = w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b

f(x) = 1 / (1 + e^(-z(x)))
```

The result is a probability, so a label is a `boolean` rather than a number: `true`
is the positive class, `false` the negative one, and the training code converts it to
`1` or `0` at the last moment (`convertBooleanToByte`). Typing it that way makes the
API refuse a "0.5 label" instead of quietly training on one.

The cost being minimised is the mean binary cross-entropy over the `m` samples:

```text
J = -(1/m) * sum over i of [ y^(i) * ln( f(x^(i)) ) + (1 - y^(i)) * ln( 1 - f(x^(i)) ) ]
```

Only one of the two terms is ever alive for a given sample, which is why
`lossFunction` branches on the label rather than evaluating both and multiplying one
of them by zero.

The partial derivatives come out looking exactly like the linear regression case's,
because the sigmoid's derivative cancels against the logarithm's:

```text
dJ/dw_j = (1/m) * sum over i of ( f(x^(i)) - y^(i) ) * x_j^(i)
dJ/db   = (1/m) * sum over i of ( f(x^(i)) - y^(i) )
```

That cancellation is the reason cross-entropy is used here and not mean squared
error: it is what keeps a confidently wrong prediction from producing a vanishing
gradient.

Each iteration updates every parameter once, from derivatives computed against the
*same* unchanged coefficients. The new weights are staged in a temporary array and
the bias is computed before that array is written back, so the update is
simultaneous.

The descent always starts at the origin — every weight `0.0`, the bias `0.0`. There
is no field in the request to move it. That is also why `J_beforeTrainScaled` is
`0.6931471805599453` for every data set: at the origin the sigmoid answers `0.5` to
everything, and `-ln 0.5` is `ln 2`.

## Feature scaling

`/train` does not descend on the payload as it arrives. `ScalingServiceImpl` divides
each column by the power of ten that makes its **first** value single-digit:

```text
r_j  = number of digits in (int) |x_j^(1)|  -  1
x'_j = x_j * 10^(-r_j)
```

A column whose first value is `55` is divided by 10, one starting at `1200` by 1000,
and one starting below 10 is left alone. The point is the learning rate: raw columns
of wildly different magnitudes carve a long narrow valley that only a tiny step size
survives, and a tiny step size needs an enormous `loopCount` to get anywhere.

Two things about it are worth knowing up front. The exponent is read off one row, so
a column whose first value is small next to the rest is barely scaled and goes on
driving the curvature. And the ratios never leave the service: `ScalingServiceImpl`
reports them, `ApiServiceImpl` drops them, and what the caller receives are
coefficients in the scaled space with no key to read them by. Both are covered under
[Known limitations](#known-limitations).

## Requirements

- **JDK 21** for an ordinary build. The POM targets release 21, and `JAVA_HOME` is
  what Maven goes by — not whatever `java` resolves to on `PATH`.
- **GraalVM for JDK 25** for a native build, plus a C toolchain. Not JDK 21; see
  [Building a native image](#building-a-native-image) for why that is not a
  preference.
- Nothing else. There is no database, no outbound call, and no configuration file to
  supply.

Docker alone is enough if you would rather not install a toolchain at all — see
[Docker](#docker) below.

## Running

```bash
./mvnw spring-boot:run
```

On Windows, `mvnw.cmd spring-boot:run`. The server listens on every interface at port
8080 — Spring Boot's own defaults, which this project does not override.

## Building a native image

```bash
./mvnw -Pnative native:compile
```

The result is a self-contained executable at `target/logistic-regression-api` that
needs no JVM to run.

Two pieces make that work, and only one of them is in this repository:

- **`native-maven-plugin`**, declared under `<build><plugins>` with no version. The
  version is managed by `spring-boot-dependencies` (1.1.8 for Spring Boot 4.1.1), so
  it moves with the Boot version rather than being pinned here.
- **The `native` profile**, which is *not* declared here — it comes from
  `spring-boot-starter-parent`. Activating it is what adds the AOT step
  (`spring-boot:process-aot`, which generates the bean definitions and reflection
  metadata that stand in for the runtime classpath scanning native-image cannot do)
  and `native:add-reachability-metadata`, which pulls in the hint files the GraalVM
  reachability metadata repository publishes for third-party libraries.

**`./mvnw -Pnative package` does not build the executable.** The parent declares
native-maven-plugin under `pluginManagement` only, so nothing binds the image build
to a lifecycle phase; naming the goal is what runs it. `native:compile` forks a
`package` run of its own first, so the AOT step still happens, and in the right
order.

### The toolchain has to be GraalVM for JDK 25

`JAVA_HOME` must point at a GraalVM, and **GraalVM for JDK 21 will not do**, even
though the bytecode this project targets is release 21. The JDK 21 line of GraalVM is
frozen on the 23.1 Substrate VM, which is too old for Spring Boot 4: it does not
understand the `typeReached` conditions the current reachability metadata is written
with, and the build then dies outright in `SerializationFeature` with a
`NoClassDefFoundError`. Measured here on Oracle GraalVM 21.0.12.1 — it fails 3.5
seconds into the image build, before any application code is looked at.

Oracle GraalVM 25.0.4.1 builds it without complaint. Only the toolchain moves; the
POM still compiles `--release 21`, so nothing about the local JDK 21 setup has to
change to keep running the application on the JVM.

Worth knowing on this machine: `JAVA_HOME` is `C:\graalvm-jdk-21.0.10+8.1`, which is
a GraalVM but the wrong one for this — a local `./mvnw -Pnative native:compile` fails
in the way described above until it points at a JDK 25 GraalVM.

**On Windows there is a second requirement.** native-image shells out to the MSVC
linker, so the build needs Visual Studio Build Tools and has to be started from an
*x64 Native Tools Command Prompt*; a normal PowerShell fails at the link step. That,
with the JDK 25 requirement above, is the main reason the [Dockerfile](#docker)
exists — it moves the whole toolchain problem into a Linux image that already has
one.

### It is memory-hungry

native-image sizes itself against the memory it can see and takes most of it.
Measured on this build: **6.29 GB of heap, 75.6 % of what the container was allowed,
peaking at 7.60 GB RSS** — against the 7.75 GB the WSL VM has in total. It fits, but
not by much. A build that dies with no error message, the process simply gone, ran
out of memory rather than hitting a compilation problem; the lever for that is the
WSL VM's allocation in `.wslconfig`, not the machine's RAM.

For reference, the image build takes **2 minutes** of native-image time on 16 cores,
and about 2 minutes 26 seconds of Maven wall clock once dependencies are cached.

### Reflection has to be told about `ScaledTrainedCoefficients`

`ApiEndpoint` carries one annotation that exists only for this:

```java
@RegisterReflectionForBinding(ScaledTrainedCoefficients.class)
```

Spring's AOT pass finds the request and response bodies through the controller and
registers them for reflection by walking their **properties** — getters, setters,
record components. The DTOs in this project expose public fields instead, so the walk
registers the outermost type's own fields and stops there: `ScaledTrainedCoefficients`,
which is only ever reached as a field of `SuccessResponseBody` and `PredictRequestBody`,
is never registered at all.

The failure is quiet rather than loud, which is what makes it worth documenting.
Without the annotation, and with no error in the log:

| | JVM | Native image |
| --- | --- | --- |
| `/train` response | `"scaledTrainedCoefficients":{"scaledWeights":[…],"scaledBias":…}` | `"scaledTrainedCoefficients":{}` |
| `/predict` | `0.8807970779778823` | `500 Unknown or Unhandled Error!!` |

With it, the two agree to the last digit. This is the shape of bug to expect from any
new DTO here: adding getters would fix it as well, and so would a
`RuntimeHintsRegistrar`, but the annotation is the smallest thing that keeps the DTO
style intact.

### What it buys, and what it costs

Measured, native image against the same code on the JVM:

| | Native image | JVM |
| --- | --- | --- |
| Startup to "Started …" | **0.036 s** | 2.113 s |
| Resident memory, idle | **31 MiB** | — |
| `/train`, 2 000 000 iterations, warm | 0.475 – 0.516 s | 0.504 – 0.564 s |
| Build time | ~2 min | seconds |
| Artifact | 107 MB executable, 186 MB image | 27 MB jar + a JRE |

Startup is the headline: roughly sixty times faster, which is the difference between
a container that is useful the moment it is scheduled and one that needs a warm-up
period.

The training loop is the interesting case, because it is exactly the kind of tight
numeric loop the JVM's JIT is supposed to win — tens of thousands of iterations over
the same code, with a profile to work from. It does not win it here. The two land
within noise of each other, with the native build slightly ahead even after the JVM
has been warmed by several full runs, and the native figures include a hop through
WSL's NAT that the JVM's do not. Boxed `Double[]` arithmetic is bound by allocation
and pointer chasing rather than by the arithmetic itself, which is the sort of code
that leaves the JIT little room to pull ahead.

So the usual AOT trade-off — better startup, worse peak throughput — is half true
here. The startup half is dramatic; the throughput half did not show up.

## Docker

The `Dockerfile` runs the native build itself, so producing the image takes nothing
but Docker — no GraalVM, no Maven, no toolchain on the host:

```bash
docker build -t bozlak/logistic-regression-api:0.1 .
```

```bash
docker run --rm -p 127.0.0.1:8080:8080 bozlak/logistic-regression-api:0.1
```

On this machine Docker runs natively inside WSL2 rather than through Docker Desktop,
so `docker` is not on the Windows `PATH`. From PowerShell:

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/javaProjects/logistic-regression-api && docker build -t bozlak/logistic-regression-api:0.1 ."
```

How it is put together, and why:

- **Two stages.** `container-registry.oracle.com/graalvm/native-image:25-ol9`
  compiles and `debian:bookworm-slim` runs. Only the executable crosses between them;
  nothing of the JDK reaches the final image, which is the whole point of compiling
  ahead of time. 186 MB against the 1.27 GB the toolchain image weighs.
- **JDK 25 and not the 21 this project targets**, because GraalVM for JDK 21 cannot
  build a Spring Boot 4 application at all — see
  [The toolchain has to be GraalVM for JDK 25](#the-toolchain-has-to-be-graalvm-for-jdk-25).
  The bytecode is still `--release 21`.
- **The `-ol9` tag is deliberate.** The default tag sits on Oracle Linux 10 and links
  against its glibc 2.39, which Debian bookworm's 2.36 cannot satisfy — the container
  would fail at the loader the moment it started. Oracle Linux 9 links against 2.34
  instead, and glibc is forward compatible, so the binary is met by bookworm and by
  anything newer. Verified: `ldd --version` reports 2.34 in the builder and 2.36 in
  the runtime.
- **Oracle GraalVM rather than the community build**, because the community images
  for JDK 25 are equally usable but Oracle's are the ones that pair with the local
  JDK 21 GraalVM already installed here. `ghcr.io/graalvm/native-image-community:25-ol9`
  is a drop-in substitute if the Oracle licence is unwelcome.
- **Dependencies are resolved on a layer of their own**, keyed only on the POM, so
  editing `src/` costs a compile rather than another walk of Maven Central. They are
  ordinary image layers rather than BuildKit cache mounts on purpose: the daemon here
  is the legacy builder — `docker buildx` is not installed — and it fails to parse
  `RUN --mount` rather than ignoring it.
- **The image ships curl but no unzip.** The Maven wrapper notices and fetches the
  `.tar.gz` distribution instead of the `.zip` on its own, so nothing has to be
  installed into the builder stage.
- **It runs as an unprivileged user** (`app`). The server reads no files and binds an
  unprivileged port, so there is nothing for root to do.
- **The port is left at Spring Boot's own default**, so the container and a local
  `./mvnw spring-boot:run` answer in the same place. Moving it is a remap —
  `-p 127.0.0.1:3002:8080` — or `-e SERVER_PORT=3002` if the change has to be visible
  inside the container.
- **Nothing has to be overridden to make the container reachable.** Spring Boot
  listens on every interface unless told otherwise, unlike the Rust services in this
  chain, whose binaries default to loopback and are unreachable in a container until
  `BIND_ADDR` says otherwise. Keep it private with the `127.0.0.1:` prefix on the host
  side of the mapping, or by not publishing the port at all.

`.dockerignore` keeps `target/` and `.git/` out of the build context. The former
reaches hundreds of megabytes as soon as the project is built locally, and every byte
of it would otherwise be handed to the daemon on each build — across the `/mnt/c`
boundary, at that.

## Configuration

There is nothing to configure for the model. What can be set are Spring Boot's own
properties, and any of them can arrive as an environment variable through relaxed
binding — `server.port` as `SERVER_PORT`, and so on:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SERVER_PORT` | `8080` | Where the server listens. The image keeps the default; remapping on the host side is usually simpler than changing it. |
| `SERVER_ADDRESS` | every interface | Which interface to bind. Setting this to `127.0.0.1` inside a container makes it unreachable from the host, which is the opposite of what it sounds like — publish to `127.0.0.1` on the host side instead. |

The application warns at startup that `/v3/api-docs` and `/swagger-ui.html` are both
enabled, and names `springdoc.api-docs.enabled` / `springdoc.swagger-ui.enabled` as
the way to turn them off. Neither turns off in the native image when passed as an
environment variable — `SPRINGDOC_SWAGGERUI_ENABLED=false` and
`SPRINGDOC_SWAGGER_UI_ENABLED=false` both leave `/swagger-ui/index.html` answering
`200`. Treat the UI as always on until that is chased down.

## API

Both endpoints hang off `/api/v1/linear-regression/use-linear-func`. The
`linear-regression` in that path is not pointing at the wrong model —
`use-linear-func` names the linear function *inside* the sigmoid, and the path
mirrors the `use_linear_func` package. It is still confusing enough from the outside
to be worth renaming one day.

### `POST /api/v1/linear-regression/use-linear-func/train`

Trains a model on the data set in the request and returns the coefficients it
converged to.

**Request** — `application/json`. `inputs[i]` is one sample's feature values and
`outputs[i]` is that sample's class:

```json
{
  "inputs": [
    [55.0,  1.0],
    [72.0,  2.0],
    [90.0,  3.0],
    [110.0, 3.0],
    [130.0, 4.0],
    [165.0, 5.0]
  ],
  "outputs": [false, false, false, true, true, true],
  "learningRate": 0.5,
  "loopCount": 20000
}
```

| Field | Accepted | Meaning |
| --- | --- | --- |
| `inputs` | non-empty, every row the same length | The samples. The feature count `n` is read from the first row. |
| `outputs` | exactly as many entries as `inputs` | `true` is the positive class. |
| `learningRate` | `(0, 1]` | Step size of each gradient descent update. |
| `loopCount` | `> 0` | How many iterations to run. There is no upper bound, and the request is held open for the whole descent — 2 000 000 iterations on the six samples above take about half a second. |

The bounds are checked before any training starts, so a bad one comes back as a `400`
rather than as a long wait.

**Response** — `200 OK`, `application/json`. This is the real reply to the request
above:

```json
{"scaledTrainedCoefficients":{"scaledWeights":[5.494790363573803,-9.139481050202411],"scaledBias":-27.587858904251725},"J_afterTrainScaled":0.0020363109289259326,"J_beforeTrainScaled":0.6931471805599453}
```

| Field | Meaning |
| --- | --- |
| `scaledTrainedCoefficients.scaledWeights` | The fit, `n` long, **in the scaled space** — see [Known limitations](#known-limitations). |
| `scaledTrainedCoefficients.scaledBias` | The bias, likewise. |
| `J_beforeTrainScaled` | Cross-entropy at the origin, where every run starts. Always `ln 2`, for every data set — it describes the starting point, not the payload. |
| `J_afterTrainScaled` | Cross-entropy the run ended on. Comparing the two is the quickest way to tell whether the learning rate was sane. |

Both readings are taken on the scaled columns — the field names say so — which makes
them comparable to each other and to another run on the same data, and to nothing
else.

### `POST /api/v1/linear-regression/use-linear-func/predict`

Scores one sample against a set of coefficients. Nothing is remembered between calls,
so the coefficients have to be sent along:

```json
{
  "scaledTrainedCoefficients": {
    "scaledWeights": [5.494790363573803, -9.139481050202411],
    "scaledBias": -27.587858904251725
  },
  "input": [11.0, 3.0]
}
```

`scaledWeights` and `input` must be the same length, or the reply is a `400`.

**Response** — `200 OK`, a bare JSON number: the sigmoid's output, between 0 and 1,
read as the probability of the positive class.

```json
0.9956637165982527
```

Note what the request does *not* carry: any indication of the space `input` is
written in. The `[11.0, 3.0]` above is the training sample `[110.0, 3.0]` after
scaling, and sending it unscaled gives a different answer — see
[Known limitations](#known-limitations).

### Errors

Every failure answers with the same shape:

```json
{"errorMessage": "..."}
```

| Status | Cause |
| --- | --- |
| `400` | `learningRate` outside `(0, 1]`, `loopCount` not positive, empty `inputs`, ragged `inputs`, an `inputs`/`outputs` length mismatch, or a `scaledWeights`/`input` length mismatch |
| `400` | A body Jackson cannot read — malformed JSON, or a field of the wrong type. The message is Jackson's own, and verbose. |
| `500` | `{"errorMessage": "Unknown or Unhandled Error!!"}` — anything else, with the real cause deliberately not disclosed. A missing or null field lands here rather than in the `400` above, because the validators dereference before they check. |

## OpenAPI

springdoc generates the schema from the controller, so the two endpoints are
browsable without a client:

- Swagger UI — <http://localhost:8080/swagger-ui.html> (a `302` to
  `/swagger-ui/index.html`)
- The document itself — <http://localhost:8080/v3/api-docs>

Both work in the native image as well as on the JVM.

## Known limitations

**The coefficients come back in the scaled space, and nothing says what the scaling
was.** `ScalingServiceImpl` reports the exponents it used as
`ScaledResult.scaleRatiosPower10`; `ApiServiceImpl.trainModel` reads only
`scaledInputs` from that result and lets the exponents fall on the floor. They appear
in no response body. A caller who wants coefficients in the units of the payload has
to rederive them from the training set by hand:

```text
w_j = w'_j * 10^(-r_j)      b = b'
```

**`/predict` does not apply them either.** It evaluates the sigmoid on `input`
exactly as sent, against the scaled weights. Chaining the two endpoints — training,
then predicting on a sample written in the same units as the training data — is
therefore only correct when the scaling happened to be a no-op, meaning every
column's first value was already below 10. Otherwise the answer is confidently wrong
rather than an error. Sample `[72.0, 2.0]` from the training set above is labelled
`false`, and against the coefficients that set produced:

| `input` sent to `/predict` | Answer | |
| --- | --- | --- |
| `[7.2, 2.0]` — pre-scaled by the caller | `0.0018250332233270053` | negative, as labelled |
| `[72.0, 2.0]` — as it appears in the data set | `1.0` | the opposite class, at full confidence |

Either pre-scale `input` the same way before sending it, or send weights lifted back
with the formula above.

**The exponent is read off one row.** `ScalingServiceImpl` takes `requestInputs[0][j]`
as the whole column's magnitude. A column whose first value is small next to the rest
is barely scaled at all and goes on driving the curvature, which is the usual reason a
learning rate that worked on one data set diverges on the next.

**The coefficients keep growing with `loopCount`.** There is no regularisation term,
so on a separable data set the weights have no finite optimum — the descent goes on
inflating them to sharpen a boundary that is already correct. The six samples above
converge to `[5.49, -9.14]` with bias `-27.59` at `learningRate` 0.5 over 20 000
iterations, and to `[10.61, -18.26]` with bias `-51.45` at 1.0 over 2 000 000, for a
cross-entropy improvement from `2.0e-3` to `1.2e-5`. The classification is identical;
only the confidence is inflated. Comparing coefficient magnitudes across runs with
different `loopCount` therefore means nothing.

**Absent fields become a `500`.** `validateRequestBody` dereferences `inputs` and
`outputs` before it checks anything, and there are no `@Valid` annotations on the
controller, so a body missing a field raises an NPE and lands in the catch-all
handler. `GlobalExceptionHandler` has a `MethodArgumentNotValidException` branch ready
for the day those annotations are added; today nothing reaches it.

**Nothing is persisted.** The service holds no model between requests by design — the
coefficients in the response are the entire result of a training run.
