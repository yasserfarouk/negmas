package exampleagentstest;

import static org.junit.Assert.assertNotNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.junit.Before;
import org.junit.Test;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.DomainImpl;
import genius.core.Global;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.exceptions.InstantiateException;
import genius.core.issue.Issue;
import genius.core.issue.IssueDiscrete;
import genius.core.issue.IssueInteger;
import genius.core.issue.IssueReal;
import genius.core.issue.Value;
import genius.core.issue.ValueInteger;
import genius.core.issue.ValueReal;
import genius.core.parties.NegotiationInfo;
import genius.core.parties.NegotiationParty;
import genius.core.persistent.PersistentDataContainer;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AbstractUtilitySpace;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * Test an example agent: try to load it, init it etc.
 */
public abstract class AgentTest {

	private NegotiationParty party;
	private Class<? extends NegotiationParty> partyclass;
	private final String PARTY = "src/test/resources/partydomain/";
	List<Class<? extends Action>> actions = Arrays.asList(Accept.class,
			Offer.class, EndNegotiation.class);
	private NegotiationInfo info;
	protected PersistentDataContainer persistentData;
	private AbstractUtilitySpace utilspace;
	private Random rand = new Random();

	public AgentTest(Class<? extends NegotiationParty> partyclass) {
		this.partyclass = partyclass;
	}

	@Before
	public void before() throws InstantiateException, IOException {
		party = (NegotiationParty) Global
				.loadObject(partyclass.getCanonicalName());

		info = mock(NegotiationInfo.class);
		when(info.getAgentID()).thenReturn(new AgentID("test"));
		persistentData = mock(PersistentDataContainer.class);
		when(info.getPersistentData()).thenReturn(persistentData);
		utilspace = createUtilSpace();
		when(info.getUtilitySpace()).thenReturn(utilspace);
		TimeLineInfo timeline = mock(TimeLineInfo.class);
		when(timeline.getTime()).thenReturn(0.2);
		when(info.getTimeline()).thenReturn(timeline);

	}

	@Test
	public void loadClassTest() throws InstantiateException {
	}

	@Test
	public void getDescriptionTest() {
		assertNotNull(party.getDescription());
	}

	@Test
	public void getProtocolTest() {
		assertNotNull(party.getProtocol());
	}

	@Test
	public void initTest() throws IOException {
		party.init(info);
	}

	@Test
	public void getActionTest() {
		party.init(info);
		party.chooseAction(actions);
	}

	@Test
	public void receiveMessageTest() {
		party.init(info);
		AgentID other = new AgentID("other");
		party.receiveMessage(other, new Offer(other, generateRandomBid()));
	}

	/**
	 * this is tricky. We need a full-fledged utilspace as the agent gets
	 * cracking on it. mocking that would be difficult.
	 * 
	 * @throws IOException
	 */
	private AbstractUtilitySpace createUtilSpace() throws IOException {
		DomainImpl domain = new DomainImpl(PARTY + "party_domain.xml");
		return new AdditiveUtilitySpace(domain, PARTY + "party1_utility.xml");
	}

	protected Bid generateRandomBid() {
		try {
			// Pairs <issue number, chosen value string>
			HashMap<Integer, Value> values = new HashMap<Integer, Value>();

			// For each issue, put a random value
			for (Issue currentIssue : utilspace.getDomain().getIssues()) {
				values.put(currentIssue.getNumber(),
						getRandomValue(currentIssue));
			}

			// return the generated bid
			return new Bid(utilspace.getDomain(), values);

		} catch (Exception e) {

			// return empty bid if an error occurred
			return new Bid(utilspace.getDomain());
		}
	}

	/**
	 * Gets a random value for the given issue.
	 *
	 * @param currentIssue
	 *            The issue to generate a random value for
	 * @return The random value generated for the issue
	 * @throws Exception
	 *             if the issues type is not Discrete, Real or Integer.
	 */
	protected Value getRandomValue(Issue currentIssue) throws Exception {

		Value currentValue;
		int index;

		switch (currentIssue.getType()) {
		case DISCRETE:
			IssueDiscrete discreteIssue = (IssueDiscrete) currentIssue;
			index = (rand.nextInt(discreteIssue.getNumberOfValues()));
			currentValue = discreteIssue.getValue(index);
			break;
		case REAL:
			IssueReal realIss = (IssueReal) currentIssue;
			index = rand.nextInt(realIss.getNumberOfDiscretizationSteps()); // check
																			// this!
			currentValue = new ValueReal(realIss.getLowerBound()
					+ (((realIss.getUpperBound() - realIss.getLowerBound()))
							/ (realIss.getNumberOfDiscretizationSteps()))
							* index);
			break;
		case INTEGER:
			IssueInteger integerIssue = (IssueInteger) currentIssue;
			index = rand.nextInt(integerIssue.getUpperBound()
					- integerIssue.getLowerBound() + 1);
			currentValue = new ValueInteger(
					integerIssue.getLowerBound() + index);
			break;
		default:
			throw new Exception(
					"issue type " + currentIssue.getType() + " not supported");
		}

		return currentValue;
	}

}
