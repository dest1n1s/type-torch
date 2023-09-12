import type { Subtract, IsPositive } from 'type-plus'

export type GreaterThanOrEqual<
  A extends number | bigint,
  B extends number | bigint,
  Fail = never
> = Subtract<A, B, 'fail'> extends infer R extends number
  ? R extends 0
    ? true
    : IsPositive<R>
  : Fail

export type NestedArray<T, D extends number = any> = GreaterThanOrEqual<D, 1> extends true
  ? D extends 1
    ? T[]
    : NestedArray<T[], Subtract<D, 1>>
  : never

export type FixedSizedArray<T, D extends number> = GreaterThanOrEqual<D, 0> extends true
  ? D extends 0
    ? []
    : [T, ...FixedSizedArray<T, Subtract<D, 1>>]
  : never

export type FixedSizedNestedArray<T, Dimensions extends number[]> = Dimensions extends []
  ? T
  : Dimensions extends [infer D, ...infer Rest]
  ? D extends number
    ? Rest extends number[]
      ? FixedSizedArray<FixedSizedNestedArray<T, Rest>, D>
      : never
    : never
  : never
