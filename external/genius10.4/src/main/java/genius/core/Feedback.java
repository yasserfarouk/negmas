package genius.core;

import java.util.List;

public enum Feedback {
	BETTER, SAME, WORSE;

	public static Vote isAcceptable(List<Feedback> feedbackList) {

		for (Feedback currentFeedback : feedbackList)
			if (currentFeedback == Feedback.WORSE)
				return Vote.REJECT;

		return Vote.ACCEPT;
	}

	public static Feedback madeupFeedback(double previous, double current) {

		if (previous > current)
			return Feedback.WORSE;
		else if (previous == current)
			return Feedback.SAME;
		else
			return Feedback.BETTER;
	}
}
