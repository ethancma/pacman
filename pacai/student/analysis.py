"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    Noise to 0 because then no unintended successor states after an action.
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    Negative 2 living reward, but not too negative or else that goes to cliff.
    Then this finds shortest path to closest exit due to penalty.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Keep same living reward from (a) for +1 closest exit. Avoid cliff by
    lowering discount.
    """

    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = -2.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    0 noise for no unintended success state.
    """

    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    kept the same, because it worked.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    kept the same, because it worked.
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    Can't discover the optimal path.
    """
    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
