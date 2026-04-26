package genius.core.analysis.pareto;

import genius.core.issue.Issue;
import genius.core.issue.Value;

/**
 * Contains issue value plus the utilities
 */
public interface IssueValue {
	/**
	 * 
	 * @return the issue that is being assigned a value
	 */
	public Issue getIssue();

	/**
	 * 
	 * @return the value of the issue
	 */
	public Value getValue();

	public Double getUtilityA();

	public Double getUtilityB();

}
