/* the frame filters shared by the list and the editor's Prev / Next */
export const FILTERS = {
    all: () => true,
    rim: it => it.has_rim,
    norim: it => !it.has_rim,
    unannotated: it => it.source === 'unannotated',
    needs_work: it => it.quality === 'needs_work',
    unusable: it => it.quality === 'unusable',
    stars: it => typeof it.quality === 'number',
    heldout: it => it.split === 'heldout',
    train: it => it.split === 'train',
    edited: it => !!it.edited,
};
