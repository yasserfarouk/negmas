package negotiator.session;

import static org.mockito.Mockito.mock;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import org.junit.After;
import org.junit.Test;

import agents.nastyagent.AddPersistentDataToStandard;
import agents.nastyagent.CheckStoredData;
import agents.nastyagent.RandomBid;
import agents.nastyagent.StoreAndRetrieve;
import agents.nastyagent.ThrowInChoose;
import agents.nastyagent.ThrowInConstructor;
import genius.core.AgentID;
import genius.core.Deadline;
import genius.core.DeadlineType;
import genius.core.events.BrokenPartyException;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionFailedEvent;
import genius.core.exceptions.InstantiateException;
import genius.core.exceptions.NegotiationPartyTimeoutException;
import genius.core.exceptions.NegotiatorException;
import genius.core.listener.Listener;
import genius.core.parties.NegotiationParty;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.parties.SessionsInfo;
import genius.core.persistent.PersistentDataType;
import genius.core.protocol.StackedAlternatingOffersProtocol;
import genius.core.repository.DomainRepItem;
import genius.core.repository.PartyRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.session.ActionException;
import genius.core.session.ExecutorWithTimeout;
import genius.core.session.RepositoryException;
import genius.core.session.Session;
import genius.core.session.SessionConfiguration;
import genius.core.session.SessionManager;

/**
 * Test if session manager correctly stores data
 *
 */
public class SessionStorageTest {

	private final static String RESOURCES = "file:src/test/resources/";

	private final String domain = RESOURCES + "partydomain/party_domain.xml";
	private final String profile = RESOURCES + "partydomain/party1_utility.xml";
	private static final Class<? extends NegotiationParty> OPPONENT = RandomBid.class;

	private DomainRepItem domainRepItem;
	private ProfileRepItem profileRepItem;
	private ExecutorWithTimeout executor = new ExecutorWithTimeout(3000);

	private Session session;
	private SessionsInfo info;

	@After
	public void after() {
		info.close();
	}

	@Test
	public void testWithStorageSerializableStorage()
			throws IOException, InstantiateException, BrokenPartyException {
		info = new SessionsInfo(new StackedAlternatingOffersProtocol(),
				PersistentDataType.SERIALIZABLE, true);
		run(StoreAndRetrieve.class);

	}

	@Test(expected = BrokenPartyException.class)
	public void testRunWithDisabledStorage()
			throws IOException, InstantiateException, BrokenPartyException {
		info = new SessionsInfo(new StackedAlternatingOffersProtocol(),
				PersistentDataType.DISABLED, true);
		run(StoreAndRetrieve.class);
	}

	@Test(expected = BrokenPartyException.class)
	public void testRunWithStandardStorage()
			throws IOException, InstantiateException, BrokenPartyException {
		info = new SessionsInfo(new StackedAlternatingOffersProtocol(),
				PersistentDataType.STANDARD, true);
		run(StoreAndRetrieve.class);
	}

	@Test
	public void checkStoreContents()
			throws IOException, InstantiateException, BrokenPartyException {
		info = new SessionsInfo(new StackedAlternatingOffersProtocol(),
				PersistentDataType.STANDARD, true);
		run(CheckStoredData.class);

	}

	@Test(expected = Exception.class) // unsupportedOperationException
	public void tryChangeStorage()
			throws IOException, InstantiateException, BrokenPartyException {
		info = new SessionsInfo(new StackedAlternatingOffersProtocol(),
				PersistentDataType.STANDARD, true);
		run(AddPersistentDataToStandard.class);

	}

	@Test(expected = InstantiateException.class)
	public void firstPartyCrashDirectly()
			throws IOException, InstantiateException, BrokenPartyException {
		info = new SessionsInfo(new StackedAlternatingOffersProtocol(),
				PersistentDataType.STANDARD, true);
		run(ThrowInConstructor.class);
	}

	@Test(expected = BrokenPartyException.class)
	public void firstPartyCrashInReceive()
			throws IOException, InstantiateException, BrokenPartyException {
		info = new SessionsInfo(new StackedAlternatingOffersProtocol(),
				PersistentDataType.STANDARD, true);
		run(ThrowInChoose.class);
	}

	/**
	 * Run session with an agent that tries to store some serializable object
	 * and that checks that the storage works
	 * 
	 * @param partyClass
	 * 
	 * @throws MalformedURLException
	 * @throws InstantiateException
	 * @throws BrokenPartyException
	 * @throws ActionException
	 * @throws InterruptedException
	 * @throws ExecutionException
	 * @throws NegotiationPartyTimeoutException
	 */
	private void run(Class<? extends NegotiationParty> partyClass)
			throws MalformedURLException, InstantiateException,
			BrokenPartyException {
		session = new Session(new Deadline(180, DeadlineType.ROUND), info);
		domainRepItem = new DomainRepItem(new URL(domain));
		profileRepItem = new ProfileRepItem(new URL(profile), domainRepItem);
		List<BrokenPartyException> errors = new ArrayList<>();

		List<NegotiationPartyInternal> theparties = generateParties(partyClass);
		SessionManager sessionMgr = new SessionManager(
				mock(SessionConfiguration.class), theparties, session,
				executor);

		sessionMgr.addListener(new Listener<NegotiationEvent>() {

			@Override
			public void notifyChange(NegotiationEvent evt) {
				if (evt instanceof SessionFailedEvent) {
					errors.add(((SessionFailedEvent) evt).getException());
				}
			}
		});
		sessionMgr.runAndWait();
		if (!errors.isEmpty()) {
			throw errors.get(0);
		}

		// run twice. 2nd time, it should check the data
		theparties = generateParties(partyClass);
		sessionMgr = new SessionManager(mock(SessionConfiguration.class),
				theparties, session, executor);

		sessionMgr.runAndWait();
		if (!errors.isEmpty()) {
			throw errors.get(0);
		}

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
