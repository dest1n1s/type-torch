import bindings from 'bindings'
const addon = bindings('type-torch')

declare class CTensor {}

export const create = (): CTensor => {
  return addon.create()
}

export const get = (tensor: CTensor) => {
  return addon.get(tensor)
}

console.log(get(create()))
