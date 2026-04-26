package genius.core.session;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.DeadlineType;
import genius.core.actions.Action;
import genius.core.events.AgentLogEvent;
import genius.core.events.BrokenPartyException;
import genius.core.events.MultipartyNegoActionEvent;
import genius.core.events.NegotiationEvent;
import genius.core.events.RecoverableSessionErrorEvent;
import genius.core.events.SessionEndedNormallyEvent;
import genius.core.events.SessionFailedEvent;
import genius.core.exceptions.NegotiationPartyTimeoutException;
import genius.core.list.Tuple;
import genius.core.listener.DefaultListenable;
import genius.core.listener.Listener;
import genius.core.parties.NegotiationParty;
import genius.core.parties.NegotiationPartyInternal;
import genius.core.persistent.PersistentDataContainer;
import genius.core.persistent.PersistentDataType;
import genius.core.protocol.MultilateralProtocol;
import genius.core.protocol.Protocol;
import genius.core.timeline.DiscreteTimeline;

/**
 * The {@link SessionManager} is responsible for enforcing the
 * {@link MultilateralProtocol} during the {@link Session}. This is the entry
 * point for the negotiation algorithm. The protocol and session parameters are
 * passed on from the GUI.
 * 
 * This logs all events to the {@link Listener}. You need to subscribe to hear
 * the log events, eg for display or for writing to file.
 * 
 *
 * @author David Festen
 * 
 */
public class SessionManager extends DefaultListenable<NegotiationEvent>
		implements Runnable {

	private static final int SESSION_ENDED_MAXTIME = 1000;

	private static final long SAVE_INFO_MAXTIME = 1000;

	private final Session session;

	// participating parties with all the session and utilspace info
	private final List<NegotiationPartyInternal> partiesInternal;

	private ExecutorWithTimeout executor;

	/**
	 * just the parties of this session. Basically a copy of
	 * {@link #partiesInternal}. Needed by the {@link Protocol}.
	 */
	private ArrayList<NegotiationParty> parties;

	/**
	 * We need to collect this, for the updates on persistentData.
	 */
	private List<Action> actions = new ArrayList<Action>();

	/**
	 * map of agent name and a short name for his profile. Needed for by
	 * {@link PersistentDataContainer}.
	 */
	private Map<String, String> profiles = new HashMap<>();

	private SessionConfiguration config;

	/**
	 * Initializes a new instance of the {@link SessionManager} object. After
	 * initialization this {@link SessionManager} can be {@link #run()}.
	 *
	 * @param theparties
	 *            The parties to use in this session (including agents and
	 *            optionally mediators)
	 * @param session
	 *            A session object containing preset information (can also be a
	 *            new instance)
	 * @param exec
	 *            the executor to use when running
	 */
	public SessionManager(SessionConfiguration config,
			List<NegotiationPartyInternal> theparties, Session session,
			ExecutorWithTimeout exec) {
		this.config = config;
		this.session = session;
		this.partiesInternal = theparties;
		this.executor = exec;

		parties = new ArrayList<NegotiationParty>();
		for (NegotiationPartyInternal p : theparties) {
			parties.add(p.getParty());
			profiles.put(p.getID().toString(),
					"Profile" + p.getUtilitySpace().getName().hashCode());
		}
	}

	/**
	 * Run and wait for completion. Can be used from a thread. Throws from the
	 * underlying call are wrapped in a {@link RuntimeException} because
	 * {@link #run()} doesn't allow throwing checked exceptions.
	 */
	public void run() {
		try {
			runAndWait();
		} catch (Exception e) {
			throw new RuntimeException("Run failed:" + e.getMessage(), e);
		}
	}

	/**
	 * Runs the negotiation session and wait for it to complete. After the run
	 * (or failure) of the session the listeners are notified. All events,
	 * success and errors are reported to the listeners. Should not throw any
	 * checked exceptions.
	 * 
	 */
	public void runAndWait() {
		// force GC to clean up mess of previous runs.
		System.gc();
		Bid agreement = null;

		try {
			executeProtocol();
			agreement = session.getInfo().getProtocol()
					.getCurrentAgreement(session, parties);
			notifyChange(new SessionEndedNormallyEvent(session, agreement,
					partiesInternal));
		} catch (Exception e) {
			notifyChange(new SessionFailedEvent(new BrokenPartyException(
					"Failed to execute protocol", config, session, e)));
		}

		// Session completed (maybe not succesfully). functions in here
		// should not throw, this is just aftermath.

		// do the agents aftermath
		callPartiesSessionEnded(agreement);
		savePartiesInfo(agreement);
	}

	/**
	 * Save the {@link PersistentDataType} of the parties if the type requires
	 * saving. If this fails, we just record an AfterSessionErrorEvent.
	 * 
	 * @throws TimeoutException
	 * @throws ExecutionException
	 */
	private void savePartiesInfo(Bid agreementBid) {
		for (final NegotiationPartyInternal party : partiesInternal) {
			final Tuple<Bid, Double> agreement = new Tuple<>(agreementBid,
					party.getUtility(agreementBid));
			try {
				new ExecutorWithTimeout(SAVE_INFO_MAXTIME).execute(
						"saving info for " + party.getID(),
						new Callable<String>() {

							@Override
							public String call() throws Exception {
								party.saveStorage(actions, profiles, agreement);
								return null;
							}
						});
			} catch (TimeoutException | ExecutionException e) {
				notifyChange(
						new RecoverableSessionErrorEvent(session, party, e));
			}

		}
	}

	/**
	 * Tell all parties {@link NegotiationParty#negotiationEnded(Bid)}.
	 * 
	 * @param agreement
	 *            the agreement bid, or null if no agreement was reached.
	 */
	private void callPartiesSessionEnded(final Bid agreement) {
		for (final NegotiationPartyInternal party : partiesInternal) {
			try {
				Map<String, String> result = new ExecutorWithTimeout(
						SESSION_ENDED_MAXTIME).execute(
								"session ended " + party.getID(),
								new Callable<Map<String, String>>() {

									@Override
									public Map<String, String> call()
											throws Exception {
										return party.getParty()
												.negotiationEnded(agreement);
									}
								});

				if (result != null && !result.isEmpty()) {
					notifyChange(new AgentLogEvent(party.getID().toString(),
							result));
				}
			} catch (ExecutionException | TimeoutException e1) {
				notifyChange(
						new RecoverableSessionErrorEvent(session, party, e1));
			}
		}
	}

	/**
	 * execute main loop (using the protocol's round structure). do
	 * before-session stuff. Then Run main loop till protocol is finished or
	 * deadline is reached. Then do after-session stuff.
	 * 
	 * @throws InvalidActionError
	 *             when a party did an invalid action
	 * @throws InterruptedException
	 *             when a party was interrupted
	 * @throws ExecutionException
	 *             when a party threw a exception
	 * @throws NegotiationPartyTimeoutException
	 *             when a party timed out
	 */
	private void executeProtocol() throws ActionException, InterruptedException,
			ExecutionException, NegotiationPartyTimeoutException {
		session.startTimer();

		handleBeforeSession();

		do {

			// generate new round
			Round round = session.getInfo().getProtocol()
					.getRoundStructure(parties, session);

			// add round to session
			session.startNewRound(round);

			if (checkDeadlineReached())
				break;
			int turnNumber = 0;

			// Let each party do an action
			for (Turn turn : round.getTurns()) {
				if (checkDeadlineReached())
					break;
				// for each party, set the round-based timeline again (to avoid
				// tempering)
				if (session.getTimeline() instanceof DiscreteTimeline) {
					((DiscreteTimeline) session.getTimeline())
							.setcRound(session.getRoundNumber());
				}

				turnNumber++;
				doPartyTurn(turnNumber, turn);

				// Do not start new turn in current round if protocol is
				// finished at this point
				if (session.getInfo().getProtocol().isFinished(session,
						parties)) {
					break;
				}
			}
			if (checkDeadlineReached())
				break;

		} while (!session.getInfo().getProtocol().isFinished(session, parties)
				&& !checkDeadlineReached());

		// stop timers if running
		if (session.isTimerRunning())
			session.stopTimer();

		// post session protocol call
		session.getInfo().getProtocol().afterSession(session, parties);
	}

	/**
	 * Handle the before-session information to be sent to the parties.
	 * 
	 * @throws NegotiationPartyTimeoutException
	 * @throws ExecutionException
	 * @throws InterruptedException
	 * @throws TimeoutException
	 */
	private void handleBeforeSession() throws NegotiationPartyTimeoutException,
			ExecutionException, InterruptedException {
		List<NegotiationParty> negoparties = new ArrayList<NegotiationParty>();
		for (NegotiationPartyInternal party : partiesInternal) {
			negoparties.add(party.getParty());
		}

		Map<NegotiationParty, List<Action>> preparatoryActions = session
				.getInfo().getProtocol().beforeSession(session, negoparties);

		for (final NegotiationParty party : preparatoryActions.keySet()) {
			for (final Action act : preparatoryActions.get(party)) {
				try {
					executor.execute(getPartyID(party).toString(),
							new Callable<Object>() {
								@Override
								public Object call() throws Exception {
									party.receiveMessage(null, act);
									return null;
								}
							});
				} catch (TimeoutException e) {
					throw new NegotiationPartyTimeoutException(party,
							"party timed out in the before-session update", e);
				}
			}

		}
	}

	/**
	 * Let a party decide for an action and create events for the taken action.
	 * 
	 * @param turnNumber
	 * @param turn
	 *            a party's {@link Turn}.
	 * @throws InvalidActionError
	 * @throws InterruptedException
	 * @throws ExecutionException
	 * @throws NegotiationPartyTimeoutException
	 */
	private void doPartyTurn(int turnNumber, Turn turn)
			throws ActionException, InterruptedException, ExecutionException,
			NegotiationPartyTimeoutException {
		NegotiationParty party = turn.getParty();
		Action action = requestAction(party, turn.getValidActions());
		turn.setAction(action);
		actions.add(action);
		updateListeners(party, action);
		notifyChange(new MultipartyNegoActionEvent(action,
				session.getRoundNumber(), session.getTurnNumber(),
				session.getTimeline().getTime(), partiesInternal,
				session.getInfo().getProtocol().getCurrentAgreement(session,
						parties)));
	}

	private boolean checkDeadlineReached() {
		// look at the time, if this is over time, remove last round and count
		// previous round
		// as most recent round
		if (session.isDeadlineReached()) {
			System.out.println("Deadline reached. " + session.getDeadlines());
			session.removeLastRound();
			if (session.getDeadlines().getType() == DeadlineType.TIME) {
				double runTimeInSeconds = (Integer) session.getDeadlines()
						.getValue();
				session.setRuntimeInSeconds(runTimeInSeconds);
			}
			return true;
		}
		return false;
	}

	/**
	 * Request an {@link Action} from the
	 * {@link genius.core.parties.NegotiationParty} given a list of valid
	 * actions and apply it according to
	 * {@link MultilateralProtocol#applyAction(Action, Session)}
	 *
	 * @param party
	 *            The party to request an action of
	 * @param validActions
	 *            the actions the party can choose
	 * @return the chosen action-
	 * @throws TimeoutException
	 */
	private Action requestAction(final NegotiationParty party,
			final List<Class<? extends Action>> validActions)
			throws ActionException, InterruptedException, ExecutionException,
			NegotiationPartyTimeoutException {

		Action action;
		try {
			action = executor.execute(getPartyID(party).toString(),
					new Callable<Action>() {
						@Override
						public Action call() throws Exception {
							ArrayList<Class<? extends Action>> possibleactions = new ArrayList<Class<? extends Action>>();
							possibleactions.addAll(validActions);
							return party.chooseAction(possibleactions);
						}
					});
		} catch (TimeoutException e) {
			String msg = "Negotiating party " + getPartyID(party)
					+ " timed out in chooseAction() method.";
			throw new NegotiationPartyTimeoutException(party, msg, e);
		}

		checkAction(party, action, validActions);
		// execute action according to protocol
		session.getInfo().getProtocol().applyAction(action, session);

		// return the chosen action
		return action;
	}

	/**
	 * Check if the action ID has been filled in properly and contains an action
	 * from the list of valid actions. No action details are checked as this is
	 * protocol dependent.
	 * 
	 * @param validActions
	 *            the allowed/valid actions
	 * 
	 * @return the given action if it is found ok
	 * @throws InvalidActionContentsError
	 * @throws InvalidActionError
	 * 
	 */
	private Action checkAction(NegotiationParty party, Action action,
			List<Class<? extends Action>> validActions) throws ActionException {
		if (action == null || action.getAgent() == null
				|| !validActions.contains(action.getClass())) {
			throw new InvalidActionError(party, validActions, action);
		}
		if (!action.getAgent().equals(getPartyID(party))) {
			throw new InvalidActionContentsError(getPartyID(party),
					"partyID " + getPartyID(party)
							+ " does not match agentID in action"
							+ action.getAgent());
		}
		return action;
	}

	/**
	 * Update all {@link NegotiationParty}s in the listeners map with the new
	 * action. Has to be here since interface (correctly) does not deal with
	 * implementation details
	 *
	 * @param actionOwner
	 *            The Party that initiated the action
	 * @param action
	 *            The action it did.
	 */
	private void updateListeners(final NegotiationParty actionOwner,
			final Action action) throws NegotiationPartyTimeoutException,
			ExecutionException, InterruptedException {
		Map<NegotiationParty, List<NegotiationParty>> listeners = session
				.getInfo().getProtocol().getActionListeners(parties);

		if (listeners == null)
			return;

		// if anyone is listening, notify any and all observers
		if (listeners.get(actionOwner) != null)
			for (final NegotiationParty observer : listeners.get(actionOwner)) {
				try {
					executor.execute(getPartyID(actionOwner).toString(),
							new Callable<Object>() {
								@Override
								public Object call() {
									observer.receiveMessage(
											getPartyID(actionOwner), action);
									return null;
								}
							});
				} catch (TimeoutException e) {
					String msg = String.format(
							"Negotiating party %s timed out in receiveMessage() method.",
							getPartyID(observer));
					throw new NegotiationPartyTimeoutException(observer, msg,
							e);
				}
			}
	}

	/**
	 * @param party
	 *            the NegotiationParty
	 * @return the {@link NegotiationPartyInternal} that contains this party.
	 */
	private NegotiationPartyInternal getNegoPartyInternal(
			NegotiationParty needed) {
		for (NegotiationPartyInternal party : partiesInternal) {
			if (party.getParty() == needed) {
				return party;
			}
		}
		throw new IllegalStateException("The referenced NegotiationParty "
				+ needed + " is not one of the actual running parties.");
	}

	/**
	 * @param party
	 *            a {@link NegotiationParty}
	 * @return the agent ID of this party.
	 */
	private AgentID getPartyID(NegotiationParty party) {
		return getNegoPartyInternal(party).getID();
	}

}
