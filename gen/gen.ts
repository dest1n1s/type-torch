import * as fs from 'fs'
import * as yaml from 'js-yaml'

const excludedFunctions = [
  'multi_margin_loss',
  'multi_margin_loss_out',
  'log_softmax_backward_data',
  'softmax_backward_data',
  'clone',
  'copy',
  'copy_out',
  'copy_',
  'conv_transpose2d_backward_out',
  'conv_transpose3d_backward_out',
  'slow_conv_transpose2d_backward_out',
  'slow_conv_transpose3d_backward_out',
  'slow_conv3d_backward_out',
  'normal',
  '_cufft_set_plan_cache_max_size',
  '_cufft_clear_plan_cache',
  'backward',
  '_amp_non_finite_check_and_unscale_',
  '_cummin_helper',
  '_cummax_helper',
  'retain_grad',
  '_validate_sparse_coo_tensor_args',
  '_backward',
  'size',
  'stride',
  '_assert_async',
  'gradient',
  'linalg_vector_norm',
  'linalg_vector_norm_out',
  'linalg_matrix_norm',
  'linalg_matrix_norm_out',
  'normal_out',
  'bernoulli_out',
  'nested_tensor',
  '_nested_tensor_from_tensor_list',
  '_nested_tensor_from_tensor_list_out'
]

const noTensorOptions = [
  'zeros_like',
  'empty_like',
  'full_like',
  'ones_like',
  'rand_like',
  'randint_like',
  'randn_like'
]

// const prefixedFunctions = [
//   'add',
//   'add_',
//   'div',
//   'div_',
//   'mul',
//   'mul_',
//   'sub',
//   'sub_',
//   'nll_loss',
//   'to_mkldnn'
// ]

const excludedPrefixes = ['_thnn_', '_th_', 'thnn_', 'th_', '_foreach', '_amp_foreach']
const excludedSuffixes = ['_forward', '_forward_out']

type Func = {
  name: string
  operator_name: string
  overload_name: string
  args: Arg[]
  returns: ReturnsType

  kind: 'function_' | 'method_'
}

type ReturnsType =
  | {
      type: 'nothing' | 'bool' | 'int64_t' | 'double' | 'dynamic'
    }
  | {
      type: 'fixed'
      length: number
    }

type ArgType =
  | 'Bool'
  | 'Int64'
  | 'Int64Option'
  | 'Double'
  | 'DoubleOption'
  | 'Tensor'
  | 'TensorOption'
  | 'IntList'
  | 'IntListOption'
  | 'DoubleList'
  | 'TensorOptList'
  | 'TensorList'
  | 'TensorOptions'
  | 'Scalar'
  | 'ScalarType'
  | 'Device'
  | 'Layout'
  | 'MemoryFormat'
  | 'String'

type Arg = {
  arg_name: string
  arg_type: ArgType
  default_value?: string
}

function getArgTypeOfString(argType: string, isNullable: boolean): ArgType | null {
  switch (argType.toLowerCase()) {
    case 'bool':
      return 'Bool'
    case 'int64_t':
      return isNullable ? 'Int64Option' : 'Int64'
    case 'double':
      return isNullable ? 'DoubleOption' : 'Double'
    case 'at::tensor':
      return isNullable ? 'TensorOption' : 'Tensor'
    case 'at::tensoroptions':
      return 'TensorOptions'
    case 'at::intarrayref':
      return isNullable ? 'IntListOption' : 'IntList'
    case 'at::arrayref<double>':
      return 'DoubleList'
    case 'const c10::list<c10::optional<at::tensor>> &':
      return 'TensorOptList'
    case 'at::tensorlist':
    case 'const at::itensorlistref &':
      return 'TensorList'
    case 'at::device':
      return 'Device'
    case 'at::layout':
      return 'Layout'
    case 'at::memoryformat':
      return 'MemoryFormat'
    case 'const at::scalar &':
    case 'at::scalar':
      return 'Scalar'
    case 'at::scalartype':
      return 'ScalarType'
    case 'c10::string_view':
      return 'String'
    default:
      return null
  }
}

function getOperatorName(func: Func) {
  if (func.operator_name === 'scatter_reduce') {
    return '_scatter_reduce'
  } else if (func.operator_name === 'scatter_reduce_') {
    return '_scatter_reduce_'
  }
  return func.operator_name
}

function getExtractedCTypeArg({ arg_name, arg_type }: Arg, index: number) {
  switch (arg_type) {
    case 'IntList':
    case 'IntListOption':
      return [
        `Napi::Array ${arg_name}_array__ = info__[${index}].As<Napi::Array>();`,
        `int ${arg_name}_len = ${arg_name}_array__.Length();`,
        `int64_t *${arg_name}_data = new int64_t[${arg_name}_len];`,
        `for (int i = 0; i < ${arg_name}_len; ++i)`,
        `  ${arg_name}_data[i] = ${arg_name}_array__.Get(i).ToNumber().Int64Value();`
      ]
    case 'DoubleList':
      return [
        `Napi::Array ${arg_name}_array__ = info__[${index}].As<Napi::Array>();`,
        `int ${arg_name}_len = ${arg_name}_array__.Length();`,
        `double *${arg_name}_data = new double[${arg_name}_len];`,
        `for (int i = 0; i < ${arg_name}_len; ++i)`,
        `  ${arg_name}_data[i] = ${arg_name}_array__.Get(i).ToNumber().DoubleValue();`
      ]
    case 'TensorOptList':
    case 'TensorList':
      return [
        `Napi::Array ${arg_name}_array__ = info__[${index}].As<Napi::Array>();`,
        `int ${arg_name}_len = ${arg_name}_array__.Length();`,
        `torch::Tensor **${arg_name}_data = new torch::Tensor *[${arg_name}_len];`,
        `for (int i = 0; i < ${arg_name}_len; ++i)`,
        `  ${arg_name}_data[i] = ${arg_name}_array__.Get(i).As<Napi::External<torch::Tensor>>().Data();`
      ]
    case 'TensorOptions':
      return [
        `int ${arg_name}_kind = info__[${index}].As<Napi::Number>().Int32Value();`,
        `int ${arg_name}_device = info__[${index + 1}].As<Napi::Number>().Int32Value();`
      ]
    case 'String':
      return [
        `Napi::String ${arg_name}_str__ = info__[${index}].As<Napi::String>();`,
        `std::string ${arg_name} = ${arg_name}_str__.Utf8Value();`
      ]
    case 'Int64Option':
      return [
        `bool ${arg_name}_null = info__[${index}].IsUndefined();`,
        `int64_t ${arg_name}_v = ${arg_name}_null ? 0 : info__[${index}].As<Napi::Number>().Int64Value();`
      ]
    case 'DoubleOption':
      return [
        `bool ${arg_name}_null = info__[${index}].IsUndefined();`,
        `double ${arg_name}_v = ${arg_name}_null ? 0 : info__[${index}].As<Napi::Number>().DoubleValue();`
      ]
    case 'Bool':
      return [`bool ${arg_name} = info__[${index}].As<Napi::Boolean>().Value();`]
    case 'Int64':
      return [`int64_t ${arg_name} = info__[${index}].As<Napi::Number>().Int64Value();`]
    case 'Double':
      return [`double ${arg_name} = info__[${index}].As<Napi::Number>().DoubleValue();`]
    case 'Tensor':
      return [
        `torch::Tensor *${arg_name} = info__[${index}].As<Napi::External<torch::Tensor>>().Data();`
      ]
    case 'TensorOption':
      return [
        `bool ${arg_name}_null = info__[${index}].IsUndefined();`,
        `torch::Tensor *${arg_name}_v = ${arg_name}_null ? nullptr : info__[${index}].As<Napi::External<torch::Tensor>>().Data();`
      ]
    case 'ScalarType':
    case 'Device':
    case 'Layout':
    case 'MemoryFormat':
      return [`int ${arg_name} = info__[${index}].As<Napi::Number>().Int32Value();`]
    case 'Scalar':
      return [
        `torch::Scalar *${arg_name} = info__[${index}].As<Napi::External<torch::Scalar>>().Data();`
      ]
    default:
      throw new Error(`Unknown arg_type: ${arg_type}`)
  }
}

function getCArgsList(args: Arg[]) {
  return args
    .map(({ arg_name, arg_type }) => {
      switch (arg_type) {
        case 'Scalar':
        case 'Tensor':
          return `*${arg_name}`
        case 'TensorOption':
          return `(${arg_name}_null ? *${arg_name}_v : torch::Tensor())`
        case 'IntList':
          return `torch::IntArrayRef(${arg_name}_data, ${arg_name}_len)`
        case 'IntListOption':
          return `${arg_name}_data == nullptr ? c10::nullopt : c10::optional<torch::IntArrayRef>(torch::IntArrayRef(${arg_name}_data, ${arg_name}_len))`
        case 'DoubleList':
          return `at::ArrayRef<double>(${arg_name}_data, ${arg_name}_len)`
        case 'TensorOptList':
          return `of_carray_tensor_opt(${arg_name}_data, ${arg_name}_len)`
        case 'TensorList':
          return `of_carray_tensor(${arg_name}_data, ${arg_name}_len)`
        case 'TensorOptions':
          return `at::device(device_of_int(${arg_name}_device)).dtype(at::ScalarType(${arg_name}_kind))`
        case 'Int64Option':
          return `${arg_name}_null ? c10::nullopt : c10::optional<int64_t>(${arg_name}_v)`
        case 'DoubleOption':
          return `${arg_name}_null ? c10::nullopt : c10::optional<double>(${arg_name}_v)`
        case 'ScalarType':
          return `at::ScalarType(${arg_name})`
        case 'Device':
          return `device_of_int(${arg_name})`
        case 'Layout':
          return `(at::Layout)${arg_name}`
        case 'MemoryFormat':
          return `(at::MemoryFormat)${arg_name}`
        default:
          return arg_name
      }
    })
    .join(', ')
}

function getCReturn(func: Func) {
  switch (func.returns.type) {
    case 'nothing':
      return 'void'
    case 'dynamic':
      return 'Napi::Array'
    case 'fixed':
      return func.returns.length === 1 ? 'Napi::External<torch::Tensor>' : 'Napi::Array'
    case 'bool':
      return 'Napi::Boolean'
    case 'int64_t':
      return 'Napi::Number'
    case 'double':
      return 'Napi::Number'
  }
}

function getCCall(func: Func) {
  if (func.kind === 'function_') {
    return `torch::${func.name}(${getCArgsList(func.args)})`
  } else {
    if (func.args.length === 0) {
      throw new Error(`Method ${func.name} has no arguments`)
    }
    const head = func.args[0]
    const tail = func.args.slice(1)
    return `${head.arg_name}->${func.name}(${getCArgsList(tail)})`
  }
}

function readYaml(filename: string) {
  const contents = fs.readFileSync(filename, 'utf8')
  const data: any[] = yaml.load(contents)
  const funcs: Func[] = data
    .map((func: any) => {
      const { name, operator_name, overload_name, deprecated, method_of, arguments: args_ } = func
      const returns: ReturnsType | null = (() => {
        if (func.returns.length === 0) {
          return { type: 'nothing' }
        } else if (func.returns.every((ret: any) => ret.dynamic_type === 'at::Tensor')) {
          return { type: 'fixed', length: func.returns.length }
        } else if (func.returns.length === 1) {
          const ret = func.returns[0]
          const returnType = ret.dynamic_type
          switch (returnType) {
            case 'bool':
              return { type: 'bool' }
            case 'int64_t':
              return { type: 'int64_t' }
            case 'double':
              return { type: 'double' }
            case 'at::TensorList" | "dynamic_type: const c10::List<c10::optional<Tensor>> &':
              return { type: 'dynamic' }
            default:
              return null
          }
        }
        return null
      })()
      const kind: 'function_' | 'method_' | null = method_of.some(
        (method: any) => method === 'namespace'
      )
        ? 'function_'
        : method_of.some((method: any) => method === 'Tensor')
        ? 'method_'
        : null
      if (
        !deprecated &&
        !excludedFunctions.includes(name) &&
        !excludedPrefixes.some((prefix) => name.startsWith(prefix)) &&
        !excludedSuffixes.some((suffix) => name.endsWith(suffix)) &&
        returns &&
        kind
      ) {
        let args: Arg[] = []
        try {
          args = args_
            .map((arg: any) => {
              const {
                name: arg_name,
                dynamic_type: arg_type,
                is_nullable,
                default: default_value
              } = arg
              const arg_type_of_string = getArgTypeOfString(arg_type, is_nullable)
              if (arg_type_of_string === 'TensorOption' && noTensorOptions.includes(name)) {
                return null
              } else if (arg_type_of_string) {
                if (arg_type_of_string === 'Scalar' && arg_name === 'self') {
                  return {
                    arg_name: 'self_scalar',
                    arg_type: 'Scalar',
                    default_value
                  }
                } else {
                  return {
                    arg_name,
                    arg_type: arg_type_of_string,
                    default_value
                  }
                }
              } else if (default_value) {
                console.log(
                  `In Processing ${name}: Unknown arg_type: ${arg_type}. Skipping argument since it has a default value.`
                )
                return null
              } else throw new Error(`Unknown arg_type: ${arg_type}`)
            })
            .filter((arg) => arg !== null)
        } catch (e) {
          console.log(`In Processing ${name}: ${e}. Skipping function.`)
          return null
        }

        return {
          name,
          operator_name,
          overload_name,
          args,
          returns,
          kind
        }
      } else return null
    })
    .filter((func) => func !== null)
  return funcs
}

const postProcess = (funcs: Func[]) => {
  const funcMultiMap = funcs
    .map<[string, Func]>((func) => [getOperatorName(func), func])
    .reduce<Record<string, Func[]>>((acc, [operatorName, func]) => {
      if (!acc[operatorName]) {
        acc[operatorName] = []
      }
      acc[operatorName].push(func)
      return acc
    }, {})
  const funcMap = Object.entries(funcMultiMap)
    .map<[string, Func][]>(([operatorName, funcs]) => {
      if (funcs.length === 1) {
        return [[operatorName, funcs[0]]]
      } else {
        const hasEmptyOverLoad = funcs.some((func) => func.overload_name === '')
        const sortedFuncs = funcs.sort((a, b) => {
          if (a.name.length !== b.name.length) {
            return a.name.length - b.name.length
          } else {
            return a.args.length - b.args.length
          }
        })
        return sortedFuncs.map<[string, Func]>((func, i) => {
          const operatorName = func.operator_name
          const overloadName = func.overload_name.toLowerCase()
          if (overloadName === '' || (i === 0 && !hasEmptyOverLoad)) {
            return [operatorName, func]
          } else if (operatorName.endsWith('_')) {
            return [`${operatorName}${overloadName}_`, func]
          } else {
            return [`${operatorName}_${overloadName}`, func]
          }
        })
      }
    })
    .reduce<Record<string, Func>>((acc, funcPairs) => {
      for (const [operatorName, func] of funcPairs) {
        acc[operatorName] = func
      }
      return acc
    }, {})
  return funcMap
}

const writeCpp = (funcs: Record<string, Func>, filename: string, directory: string) => {
  const cppFile = `${directory}/${filename}.cc`
  const hFile = `${directory}/${filename}.h`
  fs.writeFileSync(cppFile, '')
  fs.writeFileSync(hFile, '')
  const pc = (str: string) => fs.appendFileSync(cppFile, str + '\n')
  const ph = (str: string) => fs.appendFileSync(hFile, str + '\n')
  pc('// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!')
  pc('')
  pc(`#include "${filename}.h"`)
  pc('')
  pc('namespace TypeTorch {')
  pc('')
  ph('// THIS FILE IS AUTOMATICALLY GENERATED, DO NOT EDIT BY HAND!')
  ph('')
  ph('#include <napi.h>')
  ph('#include "torch_api.h"')
  ph('')
  ph('namespace TypeTorch {')
  ph('')
  Object.entries(funcs).forEach(([exportedName, func]) => {
    pc(`${getCReturn(func)} atg_${exportedName}(const Napi::CallbackInfo &info__) {`)
    ph(`${getCReturn(func)} atg_${exportedName}(const Napi::CallbackInfo &info__);`)
    pc(`  Napi::Env env__ = info__.Env();`)
    func.args.forEach((arg, index) => {
      const extractedCTypeArg = getExtractedCTypeArg(arg, index)
      extractedCTypeArg.forEach((line) => pc(`  ${line}`))
    })
    if (func.returns.type === 'nothing') {
      pc(`  ${getCCall(func)};`)
    } else {
      pc(`  auto output__ = ${getCCall(func)};`)
    }
    if (func.returns.type === 'fixed') {
      if (func.returns.length === 1) {
        pc(`  return Napi::External<torch::Tensor>::New(env__, new torch::Tensor(output__));`)
      } else {
        pc(`  Napi::Array output__array__ = Napi::Array::New(env__, ${func.returns.length});`)
        for (let i = 0; i < func.returns.length; ++i) {
          pc(
            `  output__array__[${i}u] = Napi::External<torch::Tensor>::New(env__, new torch::Tensor(std::get<${i}>(output__)));`
          )
        }
        pc(`  return output__array__;`)
      }
    } else if (func.returns.type === 'dynamic') {
      pc(`  Napi::Array output__array__ = Napi::Array::New(env__, output__.size());`)
      pc(`  for (int i = 0; i < output__.size(); ++i) {`)
      pc(
        `    output__array__[i] = Napi::External<torch::Tensor>::New(env__, new torch::Tensor(output__[i]));`
      )
      pc(`  }`)
      pc(`  return output__array__;`)
    } else if (func.returns.type !== 'nothing') {
      pc(`  return ${getCReturn(func)}::New(env__, output__);`)
    }
    pc('}')
    pc('')
  })
  pc('Napi::Object InitTypeTorchGenerated(Napi::Env env__, Napi::Object exports) {')
  ph('Napi::Object InitTypeTorchGenerated(Napi::Env env__, Napi::Object exports);')
  Object.keys(funcs).forEach((exportedName) => {
    pc(`  exports.Set("atg_${exportedName}", Napi::Function::New(env__, atg_${exportedName}));`)
  })
  pc('  return exports;')
  pc('}')
  pc('')
  ph('')
  pc('} // namespace TypeTorch')
  ph('} // namespace TypeTorch')
}

function main() {
  const funcs = readYaml('third_party/pytorch/Declarations-v2.0.0.yaml')
  const funcMap = postProcess(funcs)
  writeCpp(funcMap, 'torch_api_generated', 'csrc')
}

main()
