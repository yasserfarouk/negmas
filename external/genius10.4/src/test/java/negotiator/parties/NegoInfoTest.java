package negotiator.parties;

import static org.junit.Assert.assertNotNull;
import static org.mockito.Matchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;

import javax.swing.table.AbstractTableModel;

import org.junit.Before;
import org.junit.Test;

import genius.core.Bid;
import genius.core.Domain;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.Objective;
import genius.core.issue.ValueDiscrete;
import genius.core.utility.AdditiveUtilitySpace;
import genius.core.utility.EvaluatorDiscrete;

public class NegoInfoTest {

	private AdditiveUtilitySpace utilspace;
	private Domain domain;
	private List<Issue> issues = new ArrayList<Issue>();

	@Before
	public void before() {
		// the absolute minimum mocks
		utilspace = mock(AdditiveUtilitySpace.class);
		domain = mock(Domain.class);
		when(utilspace.getDomain()).thenReturn(domain);
	}

	@Test
	public void testConstructorNoIssues() throws Exception {
		// smoke test with minimal config.
		NegoInfo model = new NegoInfo(new Bid(domain), new Bid(domain),
				utilspace);
		checkConsistency(model);
	}

	@Test
	public void testConstructor2Issues() throws Exception {
		addTwoIssues();
		NegoInfo model = new NegoInfo(new Bid(domain), new Bid(domain),
				utilspace);
		checkConsistency(model);
	}

	/****************** support ***********************/
	private void addTwoIssues() {
		when(utilspace.getEvaluator(anyInt())).thenReturn(
				mock(EvaluatorDiscrete.class));

		issues.add(issue());
		issues.add(issue());

		Objective objectivesRoot = mock(Objective.class);
		Objective objective = mock(Objective.class);

		when(domain.getObjectivesRoot()).thenReturn(objectivesRoot);
		when(objectivesRoot.getObjective(anyInt())).thenReturn(objective);

		when(domain.getIssues()).thenReturn(issues);

	}

	private static int nr = 1;

	Issue issue() {
		IssueDiscrete issue = mock(IssueDiscrete.class);
		when(issue.getNumber()).thenReturn(nr++);
		when(issue.getValues()).thenReturn(value());
		return issue;
	}

	private List<ValueDiscrete> value() {
		ArrayList<ValueDiscrete> values = new ArrayList<ValueDiscrete>();
		for (int n = 0; n < 3; n++) {
			ValueDiscrete value = mock(ValueDiscrete.class);
			values.add(value);
		}
		return values;
	}

	private void checkConsistency(AbstractTableModel model) {
		for (int col = 0; col < model.getColumnCount(); col++) {
			assertNotNull(model.getColumnName(col));
			for (int row = 0; row < model.getRowCount(); row++) {
				assertNotNull("model value in row " + row + ", column " + col
						+ " is not set", model.getValueAt(row, col));
			}
		}
	}
}
