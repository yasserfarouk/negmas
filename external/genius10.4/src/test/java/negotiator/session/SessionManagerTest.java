package negotiator.session;

import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.mock;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import agents.nastyagent.BadBid;
import agents.nastyagent.BadSuperInit;
import agents.nastyagent.NastyAgent;
import agents.nastyagent.NearlyOutOfMem;
import agents.nastyagent.NonsenseActionInChoose;
import agents.nastyagent.NullBid;
import agents.nastyagent.OnlyBestBid;
import agents.nastyagent.OutOfMem;
import agents.nastyagent.SleepInChoose;
import agents.nastyagent.SleepInNegoEnd;
import agents.nastyagent.SleepInReceiveMessage;
import agents.nastyagent.StoreNull;
import agents.nastyagent.StoreUnserializableThing;
import agents.nastyagent.ThrowInChoose;
import agents.nastyagent.ThrowInConstructor;
import agents.nastyagent.ThrowInNegoEnd;
import agents.nastyagent.ThrowInReceiveMessage;
import agents.nastyagent.UnknownValue;
import genius.core.AgentID;
import genius.core.Deadline;
import genius.core.DeadlineType;
import genius.core.events.BrokenPartyException;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionFailedEvent;
import genius.core.exceptions.InstantiateException;
import genius.core.exceptions.NegotiatorException;
import genius.core.listener.Listener;
import genius.core.parties.NegotiationParty;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.parties.SessionsInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.protocol.MultilateralProtocol;
import genius.core.protocol.StackedAlternatingOffersProtocol;
import genius.core.repository.DomainRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.session.ExecutorWithTimeout;
import genius.core.session.RepositoryException;
import genius.core.session.Session;
import genius.core.session.SessionConfiguration;
import genius.core.session.SessionManager;

/**
 * Semi-end-to-end test. We use real {@link NegotiationParty} to test the
 * {@link SessionManager}.
 * 
 * @author W.Pasman
 *
 */
@RunWith(Parameterized.class)
public class SessionManagerTest {
	private static final Class<? extends NegotiationParty> OPPONENT = OnlyBestBid.class;

	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { { OPPONENT, null },
				{ BadBid.class, BrokenPartyException.class },
				{ BadSuperInit.class, InstantiateException.class },
				{ NearlyOutOfMem.class, BrokenPartyException.class },
				{ NonsenseActionInChoose.class, BrokenPartyException.class },
				{ NullBid.class, BrokenPartyException.class },
				{ OutOfMem.class, BrokenPartyException.class },
				{ SleepInChoose.class, BrokenPartyException.class },
				// This one seems to hang the system!
				// { SleepInConstructor.class,
				// NullPointerException.class },
				// { SleepInInit.class, NullPointerException.class },
				{ SleepInReceiveMessage.class, BrokenPartyException.class },
				{ ThrowInChoose.class, BrokenPartyException.class },
				{ ThrowInConstructor.class, InstantiateException.class },
				{ ThrowInReceiveMessage.class, BrokenPartyException.class },
				{ UnknownValue.class, null },

				{ SleepInNegoEnd.class, null }, { ThrowInNegoEnd.class, null },
				{ StoreNull.class, null },
				{ StoreUnserializableThing.class, null },

		});
	}

	final String RESOURCES = "file:src/test/resources/";
	final String domain = RESOURCES + "partydomain/party_domain.xml";
	final String profile = RESOURCES + "partydomain/party1_utility.xml";
	private DomainRepItem domainRepItem;
	private Session session;
	private MultilateralProtocol protocol = new StackedAlternatingOffersProtocol();
	private ExecutorWithTimeout executor = new ExecutorWithTimeout(3000);
	private ProfileRepItem profileRepItem;

	private Class<? extends NegotiationParty> partyClass;
	private Class<? extends Exception> expectedExceptionClass;
	private Exception actualException;
	private SessionsInfo info;

	@Before
	public void before() throws IOException {
		this.info = new SessionsInfo(null, PersistentDataType.DISABLED, true);
	}

	@After
	public void after() {
		info.close();
	}

	/**
	 * Test if running the given partyClass against {@link OnlyBestBid} gives us
	 * the correct behaviour.
	 * 
	 * @param partyClass
	 *            a class extending {@link NegotiationParty} that is to be used
	 *            as first agent in a nego.
	 * @param eClass
	 *            the class of the expected {@link Exception}
	 * @param n
	 * @throws IOException
	 */
	public SessionManagerTest(Class<? extends NegotiationParty> partyClass,
			Class<? extends Exception> eClass) throws IOException {
		this.partyClass = partyClass;
		this.expectedExceptionClass = eClass;
		System.out.println("Running " + partyClass);
	}

	@Test
	public void testRun() throws IOException {
		session = new Session(new Deadline(180, DeadlineType.ROUND),
				new SessionsInfo(protocol, PersistentDataType.DISABLED, true));
		domainRepItem = new DomainRepItem(new URL(domain));
		profileRepItem = new ProfileRepItem(new URL(profile), domainRepItem);

		try {
			List<NegotiationPartyInternal> theparties = generateParties(
					partyClass);
			SessionManager sessionMgr = new SessionManager(
					mock(SessionConfiguration.class), theparties, session,
					executor);
			sessionMgr.addListener(new Listener<NegotiationEvent>() {

				@Override
				public void notifyChange(NegotiationEvent evt) {
					if (evt instanceof SessionFailedEvent) {
						actualException = ((SessionFailedEvent) evt)
								.getException();
					}
				}
			});
			sessionMgr.runAndWait();

		} catch (Exception e) {
			actualException = e;
		}
		checkOutcome();
	}

	/**
	 * Check if always
	 * {@link NegotiationParty#negotiationEnded(genius.core.Bid)} is called
	 * after the nego (if the party was generated ok to start with).
	 * 
	 * @throws IOException
	 * 
	 * @throws InstantiateException
	 * @throws MalformedURLException
	 * 
	 */
	@Test
	public void testEndingProperly() throws IOException {
		List<NegotiationPartyInternal> theparties = new ArrayList<>();
		try {
			session = new Session(new Deadline(180, DeadlineType.ROUND),
					new SessionsInfo(protocol, PersistentDataType.DISABLED,
							true));
			domainRepItem = new DomainRepItem(new URL(domain));
			profileRepItem = new ProfileRepItem(new URL(profile),
					domainRepItem);
			theparties = generateParties(partyClass);
		} catch (MalformedURLException | InstantiateException e) {
			return; // crashes here are checked in #testRun
		}

		// if we get here, session starts. It should then always END properly
		SessionManager sessionMgr = new SessionManager(
				mock(SessionConfiguration.class), theparties, session,
				executor);
		try {
			sessionMgr.run();
		} catch (RuntimeException e) {
			// if it errs is not relevant in this test. What matters is if
			// negotioationEnded is called.
		}
		checkEndedProperly(theparties);
	}

	/**
	 * Check if all parties received the negotiationEnded call
	 * 
	 * @param theparties
	 */
	private void checkEndedProperly(List<NegotiationPartyInternal> theparties) {
		for (NegotiationPartyInternal party : theparties) {
			if (!((NastyAgent) (party.getParty())).isEnded()) {
				throw new IllegalStateException("Agent " + party
						+ " did not receive final negotiationEnded call");
			}
		}
	}

	private void checkOutcome() {
		boolean ok;
		Class<? extends Exception> actualExClass = actualException != null
				? actualException.getClass() : null;
		if (expectedExceptionClass != null) {
			ok = expectedExceptionClass.equals(actualExClass);
		} else {
			ok = null == actualException;
		}
		if (!ok) {
			System.err.println("Failure running " + partyClass + ": expected "
					+ string(expectedExceptionClass) + " but got "
					+ string(actualExClass));
			if (actualException != null) {
				actualException.printStackTrace();
			}
			assertEquals(expectedExceptionClass, actualExClass);
		}
	}

	String string(Class<? extends Exception> eClass) {
		return eClass == null ? "No exception" : eClass.toString();
	}

	private List<NegotiationPartyInternal> generateParties(
			Class<? extends NegotiationParty> partyClass)
			throws InstantiateException {
		ArrayList<NegotiationPartyInternal> parties = new ArrayList<NegotiationPartyInternal>();
		try {
			parties.add(createParty(partyClass));
			parties.add(createParty(OPPONENT));
		} catch (MalformedURLException | IllegalAccessException
				| ClassNotFoundException | RepositoryException
				| NegotiatorException | InstantiationException e) {
			throw new InstantiateException(
					"Failed to create party " + partyClass, e);
		}
		return parties;
	}

	/**
	 * Create a real party based on the class
	 * 
	 * @param partyClass
	 * @return {@link NegotiationPartyInternal}
	 * @throws InstantiateException
	 *             if party can't be instantiated
	 */
	private NegotiationPartyInternal createParty(
			Class<? extends NegotiationParty> partyClass)
			throws MalformedURLException, InstantiationException,
			IllegalAccessException, ClassNotFoundException, RepositoryException,
			NegotiatorException, InstantiateException {
		PartyRepItem partyRepItem = new PartyRepItem(
				partyClass.getCanonicalName());

		return new NegotiationPartyInternal(partyRepItem, profileRepItem,
				session, info, getAgentID(partyClass));
	}

	private AgentID getAgentID(Class<? extends NegotiationParty> partyClass) {
		return new AgentID(partyClass.getName());
	}
}
