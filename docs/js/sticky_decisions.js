/** Port of sticky_decisions.stick_p_t */

export function stick_p_t(s, t) {
  // P(a_t = a | a_0 = a) = 1/2 *(1 + (2s - 1)^t)
  return 0.5 * (1 + (2 * s - 1) ** t);
}
