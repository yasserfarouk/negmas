package genius.core.session;

import static genius.core.misc.ConsoleHelper.useConsoleOut;
import static genius.core.misc.Time.prettyTimeSpan;
import static java.lang.String.format;

import java.util.List;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;

import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.Constructor;

import genius.core.AgentID;
import genius.core.config.MultilateralTournamentConfiguration;
import genius.core.events.BrokenPartyException;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionFailedEvent;
import genius.core.events.TournamentEndedEvent;
import genius.core.events.TournamentSessionStartedEvent;
import genius.core.events.TournamentStartedEvent;
import genius.core.exceptions.InstantiateException;
import genius.core.exceptions.NegotiatorException;
import genius.core.list.Tuple;
import genius.core.listener.DefaultListenable;
import genius.core.listener.Listenable;
import genius.core.listener.Listener;
import genius.core.misc.RLBOAUtils;
import genius.core.parties.NegotiationParty;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.parties.SessionsInfo;
import genius.core.protocol.MultilateralProtocol;
import genius.core.repository.MultiPartyProtocolRepItem;
import genius.core.timeline.Timeline;
import genius.core.tournament.SessionConfigurationList;
import genius.core.tournament.TournamentConfiguration;

/**
 * Manages a multi-lateral tournament and makes sure that the
 * {@link genius.core.session.SessionManager} are instantiated. It uses the
 * configuration object which is created by the user interface and extracts
 * individual session from configuration object which it wil pass on to the
 * session manager.
 * 
 * <p>
 * Agents in a tournament must be of class {@link NegotiationParty}.
 */
public class TournamentManager extends Thread
		implements Listenable<NegotiationEvent> {

	/**
	 * Holds the configuration used by this tournament manager
	 */
	private SessionConfigurationList sessionConfigurationsList;

	/**
	 * Used to silence and restore console output for agents
	 */
	PrintStream orgOut = System.out;
	PrintStream orgErr = System.err;

	/**
	 * Used for printing time related console output.
	 */
	long start = System.nanoTime();

	/**
	 * our listeners.
	 */
	private DefaultListenable<NegotiationEvent> listeners = new DefaultListenable<>();

	private SessionsInfo info;

	/**
	 * Initializes a new instance of the
	 * {@link genius.core.session.TournamentManager} class. The tournament
	 * manager uses the provided configuration to find which sessions to run and
	 * how many collections of these sessions (tournaments) to run.
	 *
	 * @param config
	 *            The configuration to use for this Tournament
	 * @throws InstantiateException
	 * @throws IOException
	 */
	public TournamentManager(MultilateralTournamentConfiguration config)
			throws IOException, InstantiateException {
		sessionConfigurationsList = new SessionConfigurationList(config);
		info = new SessionsInfo(getProtocol(config.getProtocolItem()),
				config.getPersistentDataType(), config.isPrintEnabled());
		if (sessionConfigurationsList.size().bitLength() > 31) {
			throw new InstantiateException(
					"Configuration results in more than 2 billion runs: "
							+ sessionConfigurationsList.size());
		}

	}

	/****************** listener support *******************/
	@Override
	public void addListener(Listener<NegotiationEvent> listener) {
		listeners.addListener(listener);
	}

	@Override
	public void removeListener(Listener<NegotiationEvent> listener) {
		listeners.removeListener(listener);
	}

	/****************** manager *****************************/

	/**
	 * Runnable implementation for thread
	 */
	@Override
	public void run() {
		start = System.nanoTime();
		try {
			this.runSessions();
			System.out.println("Tournament completed");
			System.out.println("------------------");
			System.out.println("");
		} catch (Exception e) {
			e.printStackTrace();
			System.out.println("Tournament exited with an error");
			System.out.println("------------------");
			System.out.println("");

		}
		long end = System.nanoTime();
		System.out.println("Run finished in " + prettyTimeSpan(end - start));
		info.close();
	}

	/**
	 * Run all sessions in the given generator.
	 * 
	 * @throws ArithmeticException
	 *             if number of sessions does not fit in a 32 bit int
	 */
	private void runSessions() {
		int sessionNumber = 0;
		int tournamentNumber = 1;
		int totalSessions = sessionConfigurationsList.size().intValueExact();
		listeners.notifyChange(
				new TournamentStartedEvent(tournamentNumber, totalSessions));

		for (SessionConfiguration sessionInfo : sessionConfigurationsList) {
			sessionNumber++;
			listeners.notifyChange(new TournamentSessionStartedEvent(
					sessionNumber, totalSessions));

			runSingleSession1(sessionInfo);

			int nDone = totalSessions * tournamentNumber + sessionNumber;
			int nRemaining = totalSessions - nDone;
			System.out.println(format("approx. %s remaining",
					prettyTimeSpan(estimatedTimeRemaining(nDone, nRemaining))));
			System.out.println("");

		}
		listeners.notifyChange(new TournamentEndedEvent());
	}

	/**
	 * Run single session and notify listeners when we're done. No throwing
	 * should occur from here.
	 * 
	 * @param sessionInfo
	 *            the sessionInfo for the session
	 */
	private void runSingleSession1(final SessionConfiguration sessionInfo) {
		List<NegotiationPartyInternal> partyList = null;
		final Session session = new Session(sessionInfo.getDeadline(), info);
		ExecutorWithTimeout executor = new ExecutorWithTimeout(
				1000 * sessionInfo.getDeadline().getTimeOrDefaultTimeout());

		try {
			partyList = getPartyList(executor, sessionInfo, info, session);
		} catch (TimeoutException | ExecutionException e) {
			e.printStackTrace();
			listeners.notifyChange(new SessionFailedEvent(
					new BrokenPartyException("failed to construct agent ",
							sessionInfo, session, e)));
			return;// do not run any further if we don't have the agents.
		}

		runSingleSession(sessionInfo, partyList, executor);
	}

	/**
	 * Run a single session for the given parties (protocol and session are also
	 * used, but extracted from the tournament manager's configuration)
	 *
	 * @param parties
	 *            the parties to run the tournament for. Must contain at least 1
	 *            party. All parties must not be null.
	 * 
	 */
	private void runSingleSession(final SessionConfiguration config,
			final List<NegotiationPartyInternal> parties,
			final ExecutorWithTimeout executor) {
		if (parties == null || parties.isEmpty()) {
			throw new IllegalArgumentException(
					"parties list doesn't contain a party");
		}
		for (NegotiationPartyInternal party : parties) {
			if (party == null) {
				throw new IllegalArgumentException(
						"parties contains a null party:" + parties);
			}
		}

		Session session = parties.get(0).getSession();

		Timeline timeline = parties.get(0).getTimeLine();
		session.setTimeline(timeline);
		SessionManager sessionManager = new SessionManager(config, parties,
				session, executor);
		sessionManager.addListener(new Listener<NegotiationEvent>() {
			@Override
			public void notifyChange(NegotiationEvent data) {
				listeners.notifyChange(data);
			}
		});
		// When training, the RLBOA agents can listen to the sessions
		if (TournamentConfiguration.getBooleanOption("accessPartnerPreferences", false))
			RLBOAUtils.addReinforcementAgentListeners(parties, sessionManager);
		setPrinting(false, info.isPrintEnabled());
		sessionManager.runAndWait();
		setPrinting(true, info.isPrintEnabled());
	}


	/**
	 * Generate the parties involved in the next round of the tournament
	 * generator. Assumes generator.hasNext(). <br>
	 * Checks various error cases and reports accordingly. If repository fails
	 * completely, we call System.exit(). useConsoleOut is called to disable
	 * console output while running agent code. <br>
	 * 
	 * @param executor
	 *            the executor to use
	 * @param config
	 *            the {@link MultilateralSessionConfiguration} to use
	 * @param info
	 *            the global {@link SessionsInfo}.
	 * @return list of parties for next round. May return null if one or more
	 *         agents could not be created.
	 * @throws TimeoutException
	 *             if we run out of time during the construction.
	 * @throws ExecutionException
	 *             if one of the agents does not construct properly
	 */
	public static List<NegotiationPartyInternal> getPartyList(
			ExecutorWithTimeout executor,
			final MultilateralSessionConfiguration config,
			final SessionsInfo info, final Session session)
			throws TimeoutException, ExecutionException {

		final List<Participant> parties = new ArrayList<Participant>(
				config.getParties());
		if (config.getProtocol().getHasMediator()) {
			parties.add(0, config.getMediator());
		}

		// add AgentIDs
		final List<Tuple<AgentID, Participant>> partiesWithId = new ArrayList<>();
		for (Participant party : parties) {
			AgentID id = AgentID
					.generateID(party.getStrategy().getUniqueName());
			partiesWithId.add(new Tuple<>(id, party));
		}

		setPrinting(false, info.isPrintEnabled());

		List<NegotiationPartyInternal> partieslist = new ArrayList<>();

		try {
			for (Tuple<AgentID, Participant> p : partiesWithId) {
				executor.execute("init" + p, new Callable<Integer>() {
					@Override
					public Integer call()
							throws RepositoryException, NegotiatorException {
						partieslist.add(new NegotiationPartyInternal(
								p.get2().getStrategy(), p.get2().getProfile(),
								session, info, p.get1()));
						return 0;
					}
				});
			}
		} finally {
			setPrinting(true, info.isPrintEnabled());
		}
		return partieslist;

	}

	@Override
	protected Object clone() throws CloneNotSupportedException {
		return super.clone();
	}

	/**
	 * Tries to switch the console output to the preferred setting. Only
	 * possible if {@link SessionsInfo#isPrintEnabled} is true.
	 * 
	 * @param isPreferablyEnabled
	 *            true if we'd like to have print to stdout enabled, false if
	 *            preferred disabled. Typically used in tournament where much
	 *            printing is bad.
	 * @param forceEnable
	 *            true if user manually set printing enabled, overriding the
	 *            default.
	 */
	private static void setPrinting(boolean isPreferablyEnabled,
			boolean forceEnable) {
		if (forceEnable) {
			// if enabled, we ignore the setPrinting preferences so
			// that all printing stays enabled.
			return;
		}
		useConsoleOut(isPreferablyEnabled);
	}

	/**
	 * Calculate estimated time remaining using extrapolation
	 *
	 * @return estimation of time remaining in nano seconds
	 */
	private double estimatedTimeRemaining(int nSessionsDone,
			int nSessionsRemaining) {
		long now = System.nanoTime() - start;
		double res = nSessionsRemaining * now / (double) nSessionsDone;
		return res;
	}

	/**
	 * Create a new instance of the Protocol object from a
	 * {@link MultiPartyProtocolRepItem}
	 *
	 * @return the created protocol.
	 * @throws InstantiateException
	 *             if failure occurs while constructing the rep item.
	 */
	public static MultilateralProtocol getProtocol(
			MultiPartyProtocolRepItem protocolRepItem)
			throws InstantiateException {

		ClassLoader loader = ClassLoader.getSystemClassLoader();
		Class<?> protocolClass;
		try {
			protocolClass = loader.loadClass(protocolRepItem.getClassPath());

			Constructor<?> protocolConstructor = protocolClass.getConstructor();

			return (MultilateralProtocol) protocolConstructor.newInstance();
		} catch (Exception e) {
			throw new InstantiateException(
					"failed to instantiate " + protocolRepItem, e);
		}

	}

}
