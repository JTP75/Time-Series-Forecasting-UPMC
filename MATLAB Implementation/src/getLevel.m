function levels = getLevel(scores)

levels = (scores>=141) + (scores>=101) + (scores>=61) + (scores>=21) + (scores>=0);
