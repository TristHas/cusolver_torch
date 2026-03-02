#include <torch/extension.h>

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstdint>
#include <string>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT_OR_DOUBLE(x)                                                     \
  TORCH_CHECK((x).scalar_type() == at::kFloat || (x).scalar_type() == at::kDouble, \
              #x " must be float32 or float64")

static inline void check_cusolver(cusolverStatus_t status, const char* where) {
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, where, " failed with status=", static_cast<int>(status));
}

static cusolverDnHandle_t get_cusolver_handle() {
  static thread_local cusolverDnHandle_t handle = nullptr;
  if (handle == nullptr) {
    check_cusolver(cusolverDnCreate(&handle), "cusolverDnCreate");
  }
  return handle;
}

template <typename scalar_t>
struct CusolverDispatch;

template <>
struct CusolverDispatch<float> {
  static cusolverStatus_t syevd_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                       int n, float* a, int lda, float* w, int* lwork) {
    return cusolverDnSsyevd_bufferSize(h, jobz, uplo, n, a, lda, w, lwork);
  }
  static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                float* a, int lda, float* w, float* work, int lwork, int* info) {
    return cusolverDnSsyevd(h, jobz, uplo, n, a, lda, w, work, lwork, info);
  }

  static cusolverStatus_t syevj_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                       int n, float* a, int lda, float* w, int* lwork, syevjInfo_t p) {
    return cusolverDnSsyevj_bufferSize(h, jobz, uplo, n, a, lda, w, lwork, p);
  }
  static cusolverStatus_t syevj(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                float* a, int lda, float* w, float* work, int lwork, int* info, syevjInfo_t p) {
    return cusolverDnSsyevj(h, jobz, uplo, n, a, lda, w, work, lwork, info, p);
  }

  static cusolverStatus_t syevj_batched_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                               int n, float* a, int lda, float* w, int* lwork, syevjInfo_t p, int b) {
    return cusolverDnSsyevjBatched_bufferSize(h, jobz, uplo, n, a, lda, w, lwork, p, b);
  }
  static cusolverStatus_t syevj_batched(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                        float* a, int lda, float* w, float* work, int lwork, int* info, syevjInfo_t p,
                                        int b) {
    return cusolverDnSsyevjBatched(h, jobz, uplo, n, a, lda, w, work, lwork, info, p, b);
  }

  static cusolverStatus_t syevdx_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cusolverEigRange_t range,
                                        cublasFillMode_t uplo, int n, float* a, int lda, float vl, float vu, int il,
                                        int iu, int* meig, float* w, int* lwork) {
    return cusolverDnSsyevdx_bufferSize(h, jobz, range, uplo, n, a, lda, vl, vu, il, iu, meig, w, lwork);
  }
  static cusolverStatus_t syevdx(cusolverDnHandle_t h, cusolverEigMode_t jobz, cusolverEigRange_t range,
                                 cublasFillMode_t uplo, int n, float* a, int lda, float vl, float vu, int il, int iu,
                                 int* meig, float* w, float* work, int lwork, int* info) {
    return cusolverDnSsyevdx(h, jobz, range, uplo, n, a, lda, vl, vu, il, iu, meig, w, work, lwork, info);
  }
};

template <>
struct CusolverDispatch<double> {
  static cusolverStatus_t syevd_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                       int n, double* a, int lda, double* w, int* lwork) {
    return cusolverDnDsyevd_bufferSize(h, jobz, uplo, n, a, lda, w, lwork);
  }
  static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                double* a, int lda, double* w, double* work, int lwork, int* info) {
    return cusolverDnDsyevd(h, jobz, uplo, n, a, lda, w, work, lwork, info);
  }

  static cusolverStatus_t syevj_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                       int n, double* a, int lda, double* w, int* lwork, syevjInfo_t p) {
    return cusolverDnDsyevj_bufferSize(h, jobz, uplo, n, a, lda, w, lwork, p);
  }
  static cusolverStatus_t syevj(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                double* a, int lda, double* w, double* work, int lwork, int* info, syevjInfo_t p) {
    return cusolverDnDsyevj(h, jobz, uplo, n, a, lda, w, work, lwork, info, p);
  }

  static cusolverStatus_t syevj_batched_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                               int n, double* a, int lda, double* w, int* lwork, syevjInfo_t p, int b) {
    return cusolverDnDsyevjBatched_bufferSize(h, jobz, uplo, n, a, lda, w, lwork, p, b);
  }
  static cusolverStatus_t syevj_batched(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                                        double* a, int lda, double* w, double* work, int lwork, int* info,
                                        syevjInfo_t p, int b) {
    return cusolverDnDsyevjBatched(h, jobz, uplo, n, a, lda, w, work, lwork, info, p, b);
  }

  static cusolverStatus_t syevdx_buffer(cusolverDnHandle_t h, cusolverEigMode_t jobz, cusolverEigRange_t range,
                                        cublasFillMode_t uplo, int n, double* a, int lda, double vl, double vu, int il,
                                        int iu, int* meig, double* w, int* lwork) {
    return cusolverDnDsyevdx_bufferSize(h, jobz, range, uplo, n, a, lda, vl, vu, il, iu, meig, w, lwork);
  }
  static cusolverStatus_t syevdx(cusolverDnHandle_t h, cusolverEigMode_t jobz, cusolverEigRange_t range,
                                 cublasFillMode_t uplo, int n, double* a, int lda, double vl, double vu, int il,
                                 int iu, int* meig, double* w, double* work, int lwork, int* info) {
    return cusolverDnDsyevdx(h, jobz, range, uplo, n, a, lda, vl, vu, il, iu, meig, w, work, lwork, info);
  }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> eigh_cuda(
    const at::Tensor& a_in,
    bool compute_vectors,
    bool lower,
    const std::string& driver,
    double tol,
    int64_t max_sweeps,
    bool sort_eig,
    int64_t il,
    int64_t iu,
    bool copy_input,
    int64_t deterministic_mode) {
  CHECK_CUDA(a_in);
  CHECK_FLOAT_OR_DOUBLE(a_in);
  TORCH_CHECK(a_in.dim() == 2 || a_in.dim() == 3, "A must be shape (N,N) or (B,N,N)");

  at::Tensor a = a_in;
  if (!a.is_contiguous()) {
    a = a.contiguous();
  }

  const bool batched = a.dim() == 3;
  if (!batched) {
    TORCH_CHECK(a.size(0) == a.size(1), "A must be square");
    a = a.unsqueeze(0);
  } else {
    TORCH_CHECK(a.size(1) == a.size(2), "A must be square on last two dims");
  }

  if (copy_input) {
    a = a.clone();
  }

  const auto b = static_cast<int>(a.size(0));
  const auto n = static_cast<int>(a.size(1));

  int il_eff = static_cast<int>(il);
  int iu_eff = static_cast<int>(iu);
  if (driver == "syevdx") {
    if (il_eff <= 0) {
      il_eff = 1;
    }
    if (iu_eff <= 0) {
      iu_eff = n;
    }
    TORCH_CHECK(il_eff >= 1 && il_eff <= n, "il must be in [1, n]");
    TORCH_CHECK(iu_eff >= il_eff && iu_eff <= n, "iu must be in [il, n]");
  }

  auto eigvals = torch::empty({b, n}, a.options());
  auto info = torch::empty({b}, torch::dtype(torch::kInt32).device(a.device()));
  auto meig = torch::full({b}, n, torch::dtype(torch::kInt32).device(torch::kCPU));

  cusolverDnHandle_t handle = get_cusolver_handle();
  if (deterministic_mode == 1) {
    check_cusolver(cusolverDnSetDeterministicMode(handle, CUSOLVER_DETERMINISTIC_RESULTS),
                   "cusolverDnSetDeterministicMode(det)");
  } else if (deterministic_mode == 2) {
    check_cusolver(cusolverDnSetDeterministicMode(handle, CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS),
                   "cusolverDnSetDeterministicMode(non-det)");
  }

  const auto jobz = compute_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  const auto uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  if (driver == "xsyev_batched") {
    cusolverDnParams_t params = nullptr;
    check_cusolver(cusolverDnCreateParams(&params), "cusolverDnCreateParams");

    size_t ws_dev_bytes = 0;
    size_t ws_host_bytes = 0;

    if (a.scalar_type() == at::kFloat) {
      auto* a_ptr = a.data_ptr<float>();
      auto* w_ptr = eigvals.data_ptr<float>();
      check_cusolver(cusolverDnXsyevBatched_bufferSize(
                         handle,
                         params,
                         jobz,
                         uplo,
                         static_cast<int64_t>(n),
                         CUDA_R_32F,
                         a_ptr,
                         static_cast<int64_t>(n),
                         CUDA_R_32F,
                         w_ptr,
                         CUDA_R_32F,
                         &ws_dev_bytes,
                         &ws_host_bytes,
                         static_cast<int64_t>(b)),
                     "cusolverDnXsyevBatched_bufferSize");

      auto work_dev = torch::empty({static_cast<int64_t>(ws_dev_bytes)},
                                   torch::dtype(torch::kUInt8).device(a.device()));
      auto work_host = torch::empty({static_cast<int64_t>(ws_host_bytes)}, torch::dtype(torch::kUInt8));

      check_cusolver(cusolverDnXsyevBatched(
                         handle,
                         params,
                         jobz,
                         uplo,
                         static_cast<int64_t>(n),
                         CUDA_R_32F,
                         a_ptr,
                         static_cast<int64_t>(n),
                         CUDA_R_32F,
                         w_ptr,
                         CUDA_R_32F,
                         work_dev.data_ptr(),
                         ws_dev_bytes,
                         work_host.data_ptr(),
                         ws_host_bytes,
                         info.data_ptr<int>(),
                         static_cast<int64_t>(b)),
                     "cusolverDnXsyevBatched");
    } else {
      auto* a_ptr = a.data_ptr<double>();
      auto* w_ptr = eigvals.data_ptr<double>();
      check_cusolver(cusolverDnXsyevBatched_bufferSize(
                         handle,
                         params,
                         jobz,
                         uplo,
                         static_cast<int64_t>(n),
                         CUDA_R_64F,
                         a_ptr,
                         static_cast<int64_t>(n),
                         CUDA_R_64F,
                         w_ptr,
                         CUDA_R_64F,
                         &ws_dev_bytes,
                         &ws_host_bytes,
                         static_cast<int64_t>(b)),
                     "cusolverDnXsyevBatched_bufferSize");

      auto work_dev = torch::empty({static_cast<int64_t>(ws_dev_bytes)},
                                   torch::dtype(torch::kUInt8).device(a.device()));
      auto work_host = torch::empty({static_cast<int64_t>(ws_host_bytes)}, torch::dtype(torch::kUInt8));

      check_cusolver(cusolverDnXsyevBatched(
                         handle,
                         params,
                         jobz,
                         uplo,
                         static_cast<int64_t>(n),
                         CUDA_R_64F,
                         a_ptr,
                         static_cast<int64_t>(n),
                         CUDA_R_64F,
                         w_ptr,
                         CUDA_R_64F,
                         work_dev.data_ptr(),
                         ws_dev_bytes,
                         work_host.data_ptr(),
                         ws_host_bytes,
                         info.data_ptr<int>(),
                         static_cast<int64_t>(b)),
                     "cusolverDnXsyevBatched");
    }

    check_cusolver(cusolverDnDestroyParams(params), "cusolverDnDestroyParams");
    auto meig_acc = meig.data_ptr<int>();
    for (int i = 0; i < b; ++i) {
      meig_acc[i] = n;
    }
  } else {
    syevjInfo_t syevj_params = nullptr;
    if (driver == "syevj" || driver == "syevj_batched") {
      check_cusolver(cusolverDnCreateSyevjInfo(&syevj_params), "cusolverDnCreateSyevjInfo");
      check_cusolver(cusolverDnXsyevjSetTolerance(syevj_params, tol), "cusolverDnXsyevjSetTolerance");
      check_cusolver(cusolverDnXsyevjSetMaxSweeps(syevj_params, static_cast<int>(max_sweeps)), "cusolverDnXsyevjSetMaxSweeps");
      check_cusolver(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig ? 1 : 0), "cusolverDnXsyevjSetSortEig");
    } else {
      TORCH_CHECK(driver == "syevd" || driver == "syevdx",
                  "driver must be one of {'syevd', 'syevj', 'syevj_batched', 'syevdx', 'xsyev_batched'}");
    }

    at::Tensor workspace;
    int lwork = 0;

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "cusolver_eigh_workspace", [&] {
      auto* a_ptr0 = a.data_ptr<scalar_t>();
      auto* w_ptr0 = eigvals.data_ptr<scalar_t>();

      if (driver == "syevd") {
        check_cusolver(CusolverDispatch<scalar_t>::syevd_buffer(handle, jobz, uplo, n, a_ptr0, n, w_ptr0, &lwork),
                       "syevd_bufferSize");
      } else if (driver == "syevj") {
        check_cusolver(CusolverDispatch<scalar_t>::syevj_buffer(handle, jobz, uplo, n, a_ptr0, n, w_ptr0, &lwork, syevj_params),
                       "syevj_bufferSize");
      } else if (driver == "syevj_batched") {
        check_cusolver(CusolverDispatch<scalar_t>::syevj_batched_buffer(handle, jobz, uplo, n, a_ptr0, n, w_ptr0, &lwork,
                                                                        syevj_params, b),
                       "syevjBatched_bufferSize");
      } else {
        int meig_tmp = 0;
        check_cusolver(CusolverDispatch<scalar_t>::syevdx_buffer(
                           handle,
                           jobz,
                           CUSOLVER_EIG_RANGE_I,
                           uplo,
                           n,
                           a_ptr0,
                           n,
                           static_cast<scalar_t>(0),
                           static_cast<scalar_t>(0),
                           il_eff,
                           iu_eff,
                           &meig_tmp,
                           w_ptr0,
                           &lwork),
                       "syevdx_bufferSize");
      }

      workspace = torch::empty({lwork}, a.options());
    });

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "cusolver_eigh_exec", [&] {
      auto* a_ptr = a.data_ptr<scalar_t>();
      auto* w_ptr = eigvals.data_ptr<scalar_t>();
      auto* info_ptr = info.data_ptr<int>();
      auto* work_ptr = workspace.data_ptr<scalar_t>();

      const auto matrix_stride = n * n;
      const auto eig_stride = n;

      if (driver == "syevj_batched") {
        check_cusolver(CusolverDispatch<scalar_t>::syevj_batched(
                           handle,
                           jobz,
                           uplo,
                           n,
                           a_ptr,
                           n,
                           w_ptr,
                           work_ptr,
                           lwork,
                           info_ptr,
                           syevj_params,
                           b),
                       "syevjBatched");
        auto meig_acc = meig.data_ptr<int>();
        for (int i = 0; i < b; ++i) {
          meig_acc[i] = n;
        }
        return;
      }

      auto meig_acc = meig.data_ptr<int>();
      for (int i = 0; i < b; ++i) {
        auto* ai = a_ptr + i * matrix_stride;
        auto* wi = w_ptr + i * eig_stride;
        auto* inf = info_ptr + i;
        if (driver == "syevd") {
          check_cusolver(CusolverDispatch<scalar_t>::syevd(handle, jobz, uplo, n, ai, n, wi, work_ptr, lwork, inf), "syevd");
          meig_acc[i] = n;
        } else if (driver == "syevj") {
          check_cusolver(CusolverDispatch<scalar_t>::syevj(handle, jobz, uplo, n, ai, n, wi, work_ptr, lwork, inf, syevj_params),
                         "syevj");
          meig_acc[i] = n;
        } else {
          int meig_host = 0;
          check_cusolver(CusolverDispatch<scalar_t>::syevdx(
                             handle,
                             jobz,
                             CUSOLVER_EIG_RANGE_I,
                             uplo,
                             n,
                             ai,
                             n,
                             static_cast<scalar_t>(0),
                             static_cast<scalar_t>(0),
                             il_eff,
                             iu_eff,
                             &meig_host,
                             wi,
                             work_ptr,
                             lwork,
                             inf),
                         "syevdx");
          meig_acc[i] = meig_host;
        }
      }
    });

    if (syevj_params != nullptr) {
      check_cusolver(cusolverDnDestroySyevjInfo(syevj_params), "cusolverDnDestroySyevjInfo");
    }
  }

  at::Tensor eigvecs;
  if (compute_vectors) {
    // cuSOLVER assumes column-major layout while torch tensors here are row-major contiguous.
    // Return vectors in torch convention: A @ V = V @ diag(w), so transpose solver output.
    eigvecs = a.transpose(-1, -2).contiguous();
  } else {
    eigvecs = torch::empty({0}, a.options());
  }

  if (!batched) {
    eigvals = eigvals.squeeze(0);
    if (compute_vectors) {
      eigvecs = eigvecs.squeeze(0);
    }
    info = info.squeeze(0);
    meig = meig.squeeze(0);
  }

  return std::make_tuple(eigvals, eigvecs, info, meig);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("eigh_cuda",
        &eigh_cuda,
        py::arg("a"),
        py::arg("compute_vectors") = true,
        py::arg("lower") = true,
        py::arg("driver") = "syevd",
        py::arg("tol") = 1e-7,
        py::arg("max_sweeps") = 100,
        py::arg("sort_eig") = true,
        py::arg("il") = 1,
        py::arg("iu") = -1,
        py::arg("copy_input") = true,
        py::arg("deterministic_mode") = 0);
}
