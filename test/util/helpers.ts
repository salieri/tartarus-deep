export function shouldSkipSlowTests(): boolean {
  return !!process.env.TARTARUS_DEEP_SKIP_SLOW_TESTS;
}
