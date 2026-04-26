package negotiator.utility;

import static org.junit.Assert.assertEquals;

import java.math.BigInteger;

import org.junit.Test;

import genius.core.list.MathTools;

public class ComTest {
	@Test
	public void testFac() {
		assertEquals(
				new BigInteger(
						"9426890448883247745626185743057242473809693764078951663494238777294707070023223798882976159207729119823605850588608460429412647567360000000000000000000000"),
				MathTools.factorial(98));
	}

	@Test
	public void testOver() {
		assertEquals(new BigInteger("56"), MathTools.over(8, 3));
		assertEquals(new BigInteger("144773075114710515"), MathTools.over(88, 72));
		assertEquals(new BigInteger("129742056422108851102647209362723718764764760333207650"), MathTools.over(188, 72));
	}
}
