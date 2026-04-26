package list;

import static org.junit.Assert.assertEquals;

import java.math.BigInteger;
import java.util.Arrays;

import org.junit.Test;

import genius.core.list.ImArrayList;
import genius.core.list.Outer;

public class OuterTest {

	static ImArrayList<String> list1 = new ImArrayList<String>(Arrays.asList(new String[] { "a", "b" }));
	static ImArrayList<String> list2 = new ImArrayList<String>(Arrays.asList(new String[] { "c", "d" }));
	static ImArrayList<String> list3 = new ImArrayList<String>(Arrays.asList(new String[] {}));
	static ImArrayList<String> list4 = new ImArrayList<String>(Arrays.asList(new String[] { "e" }));
	static ImArrayList<String> list5 = new ImArrayList<String>(Arrays.asList(new String[] { "f", "g", "h" }));
	static ImArrayList<String> list6 = new ImArrayList<String>(Arrays.asList(new String[] { "i" }));

	@Test
	public void test() {
		Outer<String> p = new Outer<String>(list1, list2);
		assertEquals(BigInteger.valueOf(4), p.size());
		assertEquals("[[a, c],[b, c],[a, d],[b, d]]", p.toString());
		System.out.println("data=" + p + "\n");
	}

	@Test
	public void testEmpty() {

		Outer<String> p = new Outer<String>(list1, list2, list3);
		assertEquals(BigInteger.valueOf(0), p.size());
	}

	@Test
	public void testEmpty2() {
		Outer<String> p = new Outer<String>();
		assertEquals(BigInteger.valueOf(0), p.size());
	}

	@Test
	public void test2() {
		Outer<String> p = new Outer<String>(list1, list4);
		assertEquals(BigInteger.valueOf(2), p.size());
		assertEquals("[[a, e],[b, e]]", p.toString());
		System.out.println("data=" + p + "\n");
	}

	@Test
	public void test3() {
		Outer<String> p = new Outer<String>(list1, list2, list5);
		assertEquals(BigInteger.valueOf(2 * 2 * 3), p.size());
		System.out.println("data=" + p + "\n");
		assertEquals(
				"[[a, c, f],[b, c, f],[a, d, f],[b, d, f],[a, c, g],[b, c, g],[a, d, g],[b, d, g],[a, c, h],[b, c, h],[a, d, h],[b, d, h]]",
				p.toString());
	}

	@Test
	public void testSmall() {
		Outer<String> p = new Outer<String>(list4, list6);
		System.out.println("data=" + p + "\n");
		assertEquals(BigInteger.ONE, p.size());
		assertEquals("[[e, i]]", p.toString());
	}

}
