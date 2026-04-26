package list;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import genius.core.list.Tuple;

public class TupleTest {

	@Test
	public void testEqual() {
		Double d = Math.random();
		Tuple<String, Double> t1 = new Tuple<>("Agent0", d);
		Tuple<String, Double> t2 = new Tuple<>("Agent0", d);
		assertEquals(t1, t2);
	}
}
