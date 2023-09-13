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

type Shift<A extends any[]> = ((...args: A) => void) extends (...args: [A[0], ...infer R]) => void
  ? R
  : never

type GrowExpRev<A extends any[], N extends number, P extends any[][]> = A['length'] extends N
  ? A
  : [...A, ...P[0]][N] extends undefined
  ? GrowExpRev<[...A, ...P[0]], N, P>
  : GrowExpRev<A, N, Shift<P>>

type GrowExp<
  A extends any[],
  N extends number,
  P extends any[][],
  L extends number = A['length']
> = L extends N
  ? A
  : L extends 8192
  ? any[]
  : [...A, ...A][N] extends undefined
  ? GrowExp<[...A, ...A], N, [A, ...P]>
  : GrowExpRev<A, N, P>

type MapItemType<T, I> = { [K in keyof T]: I }

export type FixedSizedArray<T, D extends number> = D extends 0
  ? []
  : MapItemType<GrowExp<[0], D, []>, T>

export type FixedSizedNestedArray<T, Dimensions extends number[]> = Dimensions extends []
  ? T
  : Dimensions extends [infer D, ...infer Rest]
  ? D extends number
    ? Rest extends number[]
      ? FixedSizedArray<FixedSizedNestedArray<T, Rest>, D>
      : never
    : never
  : never
