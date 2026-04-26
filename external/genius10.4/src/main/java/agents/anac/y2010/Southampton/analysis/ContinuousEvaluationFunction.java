package agents.anac.y2010.Southampton.analysis;

import java.util.ArrayList;

public class ContinuousEvaluationFunction<T extends ContinuousEvaluationSection> {

	protected double weight;
	protected ArrayList<T> sections;

	public ContinuousEvaluationFunction(ArrayList<T> sections, double weight) {
		this.sections = sections;
		this.weight = weight;
	}

	/**
	 * @return
	 */
	public int getSectionCount() {
		return sections.size();
	}

	/**
	 * @param sectionNo
	 * @return
	 */
	public ContinuousEvaluationSection getSection(int sectionNo) {
		return sections.get(sectionNo);
	}

}