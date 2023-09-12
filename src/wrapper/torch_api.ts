import bindings from 'bindings'
import { NestedArray } from '../utils/type.js'

export declare class CTensor {}
export declare class CScalar {}

export const addon = bindings('type-torch')

export const at_new_tensor = ({
  value,
  options
}: {
  value: number
  options: { dtype: number; device: number }
}): CTensor => {
  return new addon.at_new_tensor(value, options)
}

export const at_new_tensor_from_array = ({
  values,
  options
}: {
  values: NestedArray<number>
  options: { dtype: number; device: number }
}): CTensor => {
  return new addon.at_new_tensor_from_array(values, options)
}

export const at_to_array = (tensor: CTensor): NestedArray<number> => {
  return addon.at_to_array(tensor)
}
