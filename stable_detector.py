from collections import deque, Counter

class StableBoardDetector:
    """
    Обёртка над detect_board_state, которая сглаживает мерцания/дёрганья
    между несколькими последовательными кадрами.
    """
    def __init__(self, history_size=5, min_agreement=3):
        self.history = deque(maxlen=history_size)
        self.min_agreement = min_agreement  # сколько раз подряд должно совпасть

    def update(self, raw_state: dict) -> dict:
        """
        raw_state — сырой результат detect_board_state() с одного кадра
        Возвращает сглаженное, устойчивое состояние доски.
        """
        self.history.append(raw_state)

        all_squares = set()
        for state in self.history:
            all_squares.update(state.keys())

        stable_state = {}
        for square in all_squares:
            values = [state.get(square) for state in self.history]
            values = [v for v in values if v is not None]

            if not values:
                continue

            most_common, count = Counter(values).most_common(1)[0]
            if count >= self.min_agreement:
                stable_state[square] = most_common

        return stable_state

    def reset(self):
        self.history.clear()


if __name__ == "__main__":
    # Симуляция: фигура "мерцает" между двумя классами на нескольких кадрах
    detector = StableBoardDetector(history_size=5, min_agreement=3)

    frames = [
        {"e4": "white-pawn", "e2": "white-pawn"},
        {"e4": "black-pawn", "e2": "white-pawn"},  # ложное мерцание цвета
        {"e4": "white-pawn", "e2": "white-pawn"},
        {"e4": "white-pawn"},                        # e2 пропал на кадре (ложный пропуск)
        {"e4": "white-pawn", "e2": "white-pawn"},
    ]

    for i, frame_state in enumerate(frames):
        stable = detector.update(frame_state)
        print(f"Кадр {i}: сырое={frame_state} -> устойчивое={stable}")

def sanity_check(state: dict) -> list:
    """
    Проверяет состояние доски на физическую валидность.
    Возвращает список найденных проблем (пустой список = всё ок).
    """
    issues = []

    if len(state) > 32:
        issues.append(f"Слишком много фигур: {len(state)} (максимум 32)")

    white_kings = sum(1 for p in state.values() if p == "white-king")
    black_kings = sum(1 for p in state.values() if p == "black-king")

    if white_kings == 0:
        issues.append("Нет белого короля")
    elif white_kings > 1:
        issues.append(f"Больше одного белого короля: {white_kings}")

    if black_kings == 0:
        issues.append("Нет чёрного короля")
    elif black_kings > 1:
        issues.append(f"Больше одного чёрного короля: {black_kings}")

    white_pawns = sum(1 for p in state.values() if p == "white-pawn")
    black_pawns = sum(1 for p in state.values() if p == "black-pawn")
    if white_pawns > 8:
        issues.append(f"Слишком много белых пешек: {white_pawns}")
    if black_pawns > 8:
        issues.append(f"Слишком много чёрных пешек: {black_pawns}")

    return issues
