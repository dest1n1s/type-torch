import { FixedSizedNestedArray } from '../utils/type.js'
import { CTensor, at_new_tensor, at_new_tensor_from_array } from '../wrapper/torch_api.js'

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
export class Tensor<Dimensions extends number[]> {
  _tensor: CTensor

  constructor(cTensor: CTensor) {
    this._tensor = cTensor
  }
}

export const createTensor = <Dimensions extends number[]>(
  values: FixedSizedNestedArray<number, Dimensions>,
  options: { dtype: number; device: number }
): Tensor<Dimensions> => {
  if (typeof values === 'number') return new Tensor(at_new_tensor({ value: values, options }))
  return new Tensor(at_new_tensor_from_array({ values, options }))
}
