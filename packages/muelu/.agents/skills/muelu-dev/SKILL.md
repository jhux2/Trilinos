---
name: muelu-dev
description: Use when developing the MueLu Trilinos multigrid package — tuning parallel/GPU performance, adding new input parameters via masterList.xml/update_params.sh, or registering a new class in the ETI (explicit template instantiation) system. Triggers on MueLu, masterList.xml, update_params.sh, classList, ForwardDeclaration, ETI.
---

# MueLu Developer Guide

Workflows for developing the MueLu multigrid package. All paths below are
relative to `packages/muelu`.

> This skill lives in `packages/muelu/.agents/skills/`, which is **not** one of
> opencode's default auto-scan locations. To have opencode auto-load it, add the
> directory to `skills.paths` in `opencode.json`, e.g.
> `"skills": { "paths": ["packages/muelu/.agents/skills"] }`.

---

## 1. Improving parallel / GPU performance

Source: `doc/UsersGuide/performance.tex`.

While tuning, set verbosity to `high` or `extreme` so MueLu prints aggregation
and coarsening details.

General advice:

- Choose an appropriate iterative solver (e.g. GMRES for non-symmetric). Minimize
  all-reduces — set `Use Single Reduction` in Belos if available.
- Start from the recommended settings for the problem type (`problem: type`).
- Set reasonable basics: `max levels` and `coarse: max size`, informed by fine-level
  problem size and sparsity.
- Pick a transfer-operator strategy: smoothed aggregation (`sa`) for symmetric
  problems; Petrov-Galerkin (`pg`) for non-symmetric. Try `unsmoothed` for sparser,
  lower-communication operators (iteration count rises, but solve time may drop).
- Enable implicit restrictor construction for symmetric problems:
  `transpose: use implicit`.
- Enable `rap: triple product` (triple matrix product for coarse operators);
  always faster for `unsmoothed` hierarchies, good unless operators are dense.
- Choose good level smoothers: Chebyshev (matrix-vector kernel) for SPD; for
  relaxation smoothers, tune the damping parameter first, then increase iterations.
- Fix bad coarsening by tuning `aggregation: min agg size` / `aggregation: max agg size`
  (~9 nodes/aggregate for 2D, ~27 for 3D isotropic regular meshes).
- Replace an expensive direct coarse solve with an iterative one via `coarse: type`.

On-node / GPU parallelism:

- Enable the Kokkos code path: `use kokkos refactor`.
- If using Gauss-Seidel, switch to multi-threaded Gauss-Seidel.

Distributed (MPI) parallelism:

- Enable matrix rebalancing: `repartition: enable`.
- Prefer processor-count-invariant smoothers (e.g. polynomial smoothing).
- Rebalance so the coarsest level lands on one processor for the direct solver
  (avoids extra communication).
- If using Zoltan2 `multijagged`, try the premigration option.
- Enable implicit rebalancing of P and R: `repartition: rebalance P and R`.

---

## 2. Adding a new input parameter

Sources: `doc/UsersGuide/update_params.sh`, `doc/UsersGuide/masterList.xml`,
`doc/UsersGuide/testMasterList.sh`. Requires `xsltproc`.

**Do not hand-edit `src/MueCentral/MueLu_MasterList.cpp`** — it is generated.

1. Edit `doc/UsersGuide/masterList.xml`. Add a `<parameter>` block inside the
   correct section element (`<general>`, `<aggregation>`, etc.). Minimum fields:

   ```xml
   <parameter>
     <name>my new option</name>
     <type>int</type>
     <default>1</default>
     <description>What this option does. May reference LaTeX labels.</description>
   </parameter>
   ```

   Optional: per-problem-type values (e.g. `<Poisson-3D>1</Poisson-3D>`) and ML
   compatibility tags (`<name-ML>`, `<type-ML>`, `<default-ML>`,
   `<compatibility-ML>`, `<description-ML>`).

2. From `doc/UsersGuide/`, run `./update_params.sh`. This uses xsltproc to
   regenerate `src/MueCentral/MueLu_MasterList.cpp` plus the doc tables
   (`paramlist*.tex`, `options_*.tex`, `mloptions_*.tex`).

3. Only if the parameter needs value translation (like `verbosity` or
   `cycle type`), add a shortcut in the `interpretParameterName` function in
   `update_params.sh`.

4. Verify with `./testMasterList.sh`: it regenerates and diffs against repo
   copies; any mismatch means the committed generated files are stale.

---

## 3. Adding a new file/class to the ETI system

Sources: `src/Utils/ClassList/`, `src/Utils/ExplicitInstantiation/`,
`src/Utils/ForwardDeclaration/` (see also its `README`).

1. Add the class name to the correct classList in `src/Utils/ClassList/`:
   - `SC-LO-GO-NO.classList` — templated on `<Scalar, LocalOrdinal, GlobalOrdinal, Node>`.
   - `LO-GO-NO.classList` — templated on ordinals + node only (no scalar).
   - `Non-Templated.classList` — not templated.

   Append an optional build guard after a ` - ` separator, e.g.:

   ```
   AMGXOperator - #if defined(HAVE_MUELU_AMGX)
   ```

2. If the class is an interface / should not be instantiated, also add its name
   to `src/Utils/ClassList/EI-Exceptions.classList`.

3. From `src/Utils/ClassList/`, run `./gen_all.sh`. It regenerates all derived
   artifacts (short-name headers, forward declarations, ETI cmake class lists)
   and touches `src/CMakeLists.txt` if new files were created.

4. Rerun CMake in your build directory so the generated `.hpp` files appear in
   the build tree.

### File map

| File / script | Role |
| --- | --- |
| `src/Utils/ClassList/*.classList` | Master lists of classes by template signature. Edit these. |
| `src/Utils/ClassList/EI-Exceptions.classList` | Classes to skip for ETI (interfaces, etc.). |
| `src/Utils/ClassList/gen_all.sh` | Runs all generators below; touches CMakeLists.txt for new files. |
| `src/Headers/gen_UseShortNames.sh` | Generates `MueLu_UseShortNames{Ordinal,Scalar}.hpp`. |
| `src/Utils/ForwardDeclaration/gen_fwdFiles.sh` | Generates `MueLu_<Class>_fwd.hpp` from `*.tmpl`. |
| `src/Utils/ExplicitInstantiation/genClassLists.sh` | Generates `ETI_{LO_GO_NO,SC_LO_GO_NO}_classes.cmake`. |
| `src/Utils/ExplicitInstantiation/gen_cppFiles.sh` | Generates per-class ETI `MueLu_<Class>.cpp` from `*.tmpl`. |
| `*.tmpl` (both dirs) | Templates expanded via `$TMPL_CLASS`, build-condition, and (fwd) `$TMPL_UPPERCASECLASS`. |

The ETI `.cpp` templates include `MueLu_<Class>_def.hpp`, define
`MUELU_ETI_GROUP(...)`, and pull in `MueLu_ETI_4arg.hpp` (SC-LO-GO-NO) or
`MueLu_ETI_3arg.hpp` (LO-GO-NO) to instantiate across the enabled type set.

---

## 4. Adding a unit test

Unit tests live in `test/unit_tests/` (and `test/unit_tests_kokkos/`). They use
the Teuchos unit test harness and are templated on `<Scalar, LocalOrdinal,
GlobalOrdinal, Node>` so they run across all enabled instantiations.

1. Create a test source file, typically one per class under test
   (e.g. `test/unit_tests/MyFactory.cpp`). Model it on an existing test such as
   `AmalgamationFactory.cpp`. Each test case is a
   `TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(TestGroup, TestName, Scalar, LocalOrdinal,
   GlobalOrdinal, Node)` block in `namespace MueLuTests`.

2. Inside each test body:
   - `#include <MueLu_UseShortNames.hpp>` for the short type aliases (`Matrix`,
     `Map`, `RCP`, etc.).
   - `MUELU_TESTING_SET_OSTREAM;` and
     `MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);` to set up output
     and skip unsupported type combinations.
   - Use `TestHelpers::Parameters::getDefaultComm()` /
     `getLib()` and `TestHelpers::TestFactory<...>::BuildMatrix(...)` to build
     test fixtures.
   - Assert with Teuchos macros: `TEST_EQUALITY`, `TEST_INEQUALITY`,
     `TEST_ASSERT`, `TEST_FLOATING_EQUALITY`, etc.

3. At the bottom of the file, register every test case for instantiation:

   ```cpp
   #define MUELU_ETI_GROUP(Scalar, LO, GO, Node) \
     TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(MyFactory, TestName, Scalar, LO, GO, Node)

   #include <MueLu_ETI_4arg.hpp>
   ```

   List one `TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT` line per test case (use `\`
   line continuations). This instantiates across the enabled type set; the
   shared `MueLu_UnitTests.cpp` provides `main`.

4. Register the source in `test/unit_tests/CMakeLists.txt`: add the file to the
   appropriate `APPEND_SET(SOURCES ...)` list (or a specialized list such as
   `SOURCES_BLOCKED`, `SOURCES_INTREPID2`, or a TPL-guarded `IF(...)` block).
   The file is compiled into the relevant `TRIBITS_ADD_EXECUTABLE` (e.g.
   `UnitTests`) and run by its `TRIBITS_ADD_TEST`.

5. If the test needs input data files, copy them to the binary dir with
   `TRIBITS_COPY_FILES_TO_BINARY_DIR` (see the `TestMatrices` / `UserData`
   examples).

6. Rerun CMake, build the `MueLu_UnitTests` executable, and run via `ctest` or
   directly (e.g. `./MueLu_UnitTests --linAlgebra=Tpetra`). A pass prints
   `End Result: TEST PASSED`. Narrow to one case with
   `--group-name=MyFactory --test-name=TestName`.

### File map

| File | Role |
| --- | --- |
| `test/unit_tests/<Name>.cpp` | A test suite: templated `..._DECL` cases + `MUELU_ETI_GROUP` instantiation block. |
| `test/unit_tests/MueLu_UnitTests.cpp` | Shared `main` (runs `Teuchos::UnitTestRepository`). Do not add test cases here. |
| `test/unit_tests/MueLu_TestHelpers.hpp` | `TestHelpers::Parameters`, `TestFactory`, fixture builders, testing macros. |
| `test/unit_tests/MueLu_Test_ETI.hpp` | ETI driver for non-unit tests (`Automatic_Test_ETI`), not the unit-test harness. |
| `test/unit_tests/CMakeLists.txt` | Registers sources into executables and `TRIBITS_ADD_TEST` entries. |
