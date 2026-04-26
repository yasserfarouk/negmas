package agents.rlboa;

import java.util.Objects;

public class State extends AbstractState implements BinnedRepresentation {
	
	private int myBin;
	private int oppBin;
	private int time;
	
	public static State TERMINAL = new State(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE);
	public static State INITIAL = new State(Integer.MIN_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE);

	// TODO: Initialize these values from config file
	private static int initialActionSpace = 10;
	private static int standardActionSpace = 3;
	
	public State(int myBin, int oppBin, int time) {
		this.myBin = myBin;
		this.oppBin = oppBin;
		this.time = time;
	}
	
	public int getActionSize() {
		if (this.getMyBin() < 0) {
			return initialActionSpace;
		}
		else {
			return standardActionSpace;
		}
	}

	public int getMyBin() {
		return this.myBin;
	}
	
	public int getOppBin() {
		return this.oppBin;
	}
	
	public int getTime() {
		return this.time;
	}
	
	public boolean isInitialState() {
		return this.equals(State.INITIAL);
	}
	
	public boolean isTerminalState() {
		return this.equals(State.TERMINAL);
	}
	
	public int hash() {
		return Objects.hash(this.getMyBin(), this.getOppBin(), this.getTime());
	}
	
	@Override
	public String toString() {
		return String.format("My bin: %d, Opp bin: %d, Time: %d", this.getMyBin(), this.getOppBin(), this.getTime());
	}
}
