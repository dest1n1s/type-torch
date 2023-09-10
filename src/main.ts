import bindings from 'bindings'
const addon = bindings('type-torch')

declare class Tensor<Dimensions extends number[]> {}

export const create = (): Tensor<[2, 2]> => {
  return addon.create()
}

export const get = <Dimensions extends number[]>(tensor: Tensor<Dimensions>): Dimensions => {
  return addon.get(tensor)
}

console.log(get(create()))
